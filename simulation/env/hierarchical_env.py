# simulation/env/hierarchical_env.py
"""
Hierarchical environment step function for two-stage agent.

Wraps ContainerTerminalEnv and provides step_hierarchical method
that handles the two-stage decision process.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from numpy.typing import NDArray

from simulation.core.facilities.yard import BooleanStorageYard, PlacementResult
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck
from simulation.core.enums import MoveType

from simulation.operations.hierarchical_moves import HierarchicalMoveGenerator
from simulation.rl.features.featurizers import (
    MoveableContainer, Destination, ParkingAction,
    SourceType, DestinationType
)
from simulation.rl.agents.hierarchical_dqn_agent import (
    HierarchicalDQNAgent, ActionPool, Stage1Selection, HierarchicalTransition
)


@dataclass
class HierarchicalStepResult:
    """Result of one hierarchical step."""
    next_state: NDArray[np.float32]
    reward: float
    done: bool
    info: Dict[str, Any]
    
    # For training
    transition: Optional[HierarchicalTransition] = None
    was_valid_move: bool = False


class HierarchicalEnvWrapper:
    """
    Wrapper that adds hierarchical stepping to ContainerTerminalEnv.
    
    This handles:
    - Multi-crane sequential decisions
    - Two-stage selection (container → destination)
    - Retry logic for invalid selections
    - Parking actions as direct actions
    """
    
    def __init__(
        self,
        env,  # ContainerTerminalEnv
        max_retries: int = 1,
        no_destination_penalty: float = -1.0
    ):
        """
        Initialize wrapper.
        
        Args:
            env: The underlying ContainerTerminalEnv
            max_retries: Max retries per timestep when selection has no destinations
            no_destination_penalty: Penalty when container has no valid destinations
        """
        self.env = env
        self.max_retries = max_retries
        self.no_destination_penalty = no_destination_penalty
        
        # Create hierarchical move generator
        self.move_gen = HierarchicalMoveGenerator(
            yard=env.yard,
            rail=env.rail,
            parking=env.parking,
            proximity=5
        )
        
        # Track train heat bays for destination scoring
        self._train_heat_bays: Set[int] = set()
    
    def reset(self, *args, **kwargs):
        """Reset underlying environment."""
        result = self.env.reset(*args, **kwargs)
        self._update_train_heat()
        return result
    
    def _update_train_heat(self):
        """Update train heat bays from current trains."""
        self._train_heat_bays.clear()
        for train_id in self.env.trains:
            anchor = self.env.rail.get_anchor_bay(train_id)
            if anchor is not None:
                self._train_heat_bays.add(anchor)
    
    def step_hierarchical(
        self,
        agent: HierarchicalDQNAgent,
        crane_id: int = 0
    ) -> HierarchicalStepResult:
        """
        Execute one hierarchical step for a single crane.
        
        Flow:
        1. Build action pool (containers + parking)
        2. Stage 1: Agent selects container or parking
        3. If parking: execute and return
        4. If container: compute destinations
        5. If no destinations: penalty, retry (same timestep)
        6. Stage 2: Agent selects destination
        7. Execute move, compute reward
        8. Return result with transition for training
        
        Args:
            agent: The hierarchical DQN agent
            crane_id: Which crane is deciding (for multi-crane)
            
        Returns:
            HierarchicalStepResult
        """
        now = self.env.current_time
        info: Dict[str, Any] = {
            "crane_id": crane_id,
            "retries": 0,
            "executed": [],
        }
        
        # Clear agent's state cache for new decision
        agent.clear_state_cache()
        
        # Get current state
        state_np = self.env.encoder.encode_with_forecast(
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks,
            self.env.day_plan,
            now
        )
        
        # Build action pool
        containers = self.move_gen.list_moveable_containers(
            self.env.trains,
            self.env.trucks,
            now
        )
        parkings = self.move_gen.list_parking_actions(self.env.trucks)
        
        pool = ActionPool(containers=containers, parkings=parkings)
        
        # Handle empty pool
        if pool.is_empty():
            self._advance_time()
            next_state = self.env.encoder.encode_with_forecast(
                self.env.trains,
                self.env.trucks,
                self.env.terminal_trucks,
                self.env.day_plan,
                self.env.current_time
            )
            done = self._check_day_end()
            return HierarchicalStepResult(
                next_state=next_state,
                reward=0.0,
                done=done,
                info=info,
                transition=None,
                was_valid_move=False
            )
        
        # Retry loop for invalid container selections
        excluded_container_ids: Set[str] = set()
        total_penalty = 0.0
        
        for retry in range(self.max_retries):
            info["retries"] = retry
            
            # Filter out excluded containers
            filtered_containers = [
                c for c in containers 
                if c.container_id not in excluded_container_ids
            ]
            filtered_pool = ActionPool(
                containers=filtered_containers,
                parkings=parkings
            )
            
            if filtered_pool.is_empty():
                # All options exhausted
                break
            
            # Stage 1: Select action
            selection = agent.select_stage1(state_np, filtered_pool)
            
            if selection.index < 0:
                # No valid selection
                break
            
            # Handle parking action
            if selection.is_parking:
                return self._execute_parking(
                    state_np, selection, filtered_pool, agent, info
                )
            
            # Handle container selection
            cont = selection.container
            
            # Stage 2: Get destinations
            destinations = self.move_gen.list_destinations_for_container(
                cont,
                self.env.trains,
                self.env.trucks,
                self._train_heat_bays
            )
            
            if not destinations:
                # No valid destinations - penalty and retry
                total_penalty += self.no_destination_penalty
                excluded_container_ids.add(cont.container_id)
                continue
            
            # Stage 2: Select destination
            dest_idx = agent.select_stage2(
                state_np,
                selection.container_feat,
                destinations,
                cont.bay,
                cont.tier
            )
            
            if dest_idx < 0 or dest_idx >= len(destinations):
                # Invalid destination selection
                total_penalty += self.no_destination_penalty
                excluded_container_ids.add(cont.container_id)
                continue
            
            dest = destinations[dest_idx]
            
            # Execute the move
            result = self._execute_move(
                state_np, cont, selection, dest, dest_idx, destinations,
                filtered_pool, agent, info
            )
            result.reward += total_penalty
            return result
        
        # All retries exhausted - advance time
        self._advance_time()
        next_state = self.env.encoder.encode_with_forecast(
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks,
            self.env.day_plan,
            self.env.current_time
        )
        done = self._check_day_end()
        
        return HierarchicalStepResult(
            next_state=next_state,
            reward=total_penalty,
            done=done,
            info=info,
            transition=None,
            was_valid_move=False
        )
    
    def _execute_parking(
        self,
        state_np: np.ndarray,
        selection: Stage1Selection,
        pool: ActionPool,
        agent: HierarchicalDQNAgent,
        info: Dict
    ) -> HierarchicalStepResult:
        """Execute parking action."""
        parking = selection.parking
        truck = self.env.trucks.get(parking.truck_id)
        
        if truck and self.env.parking:
            success = self.env.parking.allocate(truck, parking.spot)
        else:
            success = False
        
        if success:
            reward = 0.5  # Parking reward from existing system
            info["executed"].append({
                "type": "PARKING",
                "truck_id": parking.truck_id,
                "spot": parking.spot
            })
        else:
            reward = -0.5
        
        # Get next state
        next_state = self.env.encoder.encode_with_forecast(
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks,
            self.env.day_plan,
            self.env.current_time
        )
        done = self._check_day_end()
        
        # Build transition for training
        parking_feats = agent.park_featurizer.featurize_batch(pool.parkings).numpy()
        
        transition = HierarchicalTransition(
            state=state_np.copy(),
            container_feats=np.zeros((0, 16), dtype=np.float32),
            container_idx=-1,
            container_feat=np.zeros(16, dtype=np.float32),
            destination_feats=np.zeros((0, 12), dtype=np.float32),
            destination_idx=-1,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            was_parking=True,
            parking_feats=parking_feats,
            parking_idx=selection.index
        )
        
        return HierarchicalStepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
            transition=transition,
            was_valid_move=success
        )
    
    def _execute_move(
        self,
        state_np: np.ndarray,
        cont: MoveableContainer,
        selection: Stage1Selection,
        dest: Destination,
        dest_idx: int,
        destinations: List[Destination],
        pool: ActionPool,
        agent: HierarchicalDQNAgent,
        info: Dict
    ) -> HierarchicalStepResult:
        """Execute container move."""
        # Determine move type based on source and destination
        move_type = self._determine_move_type(cont.source_type, dest.dest_type)
        
        # Build move args
        args = self._build_move_args(cont, dest)
        
        # Execute through TLM
        success = self.env.tlm.execute(
            type(self.env.tlm).Move(move_type, args) if hasattr(self.env.tlm, 'Move') 
            else self._create_move(move_type, args),
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks
        )
        
        if success:
            # Compute reward
            reward = self.env.reward_engine.immediate_reward(
                move_type.value,
                distance_m=abs(dest.bay - cont.bay) * 2.0,  # Approximate distance
                time_s=30.0  # Approximate time
            )
            info["executed"].append({
                "type": move_type.value,
                "container_id": cont.container_id,
                "source": cont.source_type.value,
                "dest": dest.dest_type.value
            })
            
            # Update train heat
            self._update_train_heat()
        else:
            reward = -0.5
        
        # Get next state
        next_state = self.env.encoder.encode_with_forecast(
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks,
            self.env.day_plan,
            self.env.current_time
        )
        done = self._check_day_end()
        
        # Build transition
        cont_feats = agent.cont_featurizer.featurize_batch(pool.containers).numpy()
        dest_feats = agent.dest_featurizer.featurize_batch(
            destinations, cont.bay, cont.tier
        ).numpy()
        
        transition = HierarchicalTransition(
            state=state_np.copy(),
            container_feats=cont_feats,
            container_idx=selection.index,
            container_feat=selection.container_feat.copy(),
            destination_feats=dest_feats,
            destination_idx=dest_idx,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            was_parking=False
        )
        
        return HierarchicalStepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
            transition=transition,
            was_valid_move=success
        )
    
    def _determine_move_type(
        self,
        source: SourceType,
        dest: DestinationType
    ) -> MoveType:
        """Map source/destination combination to MoveType."""
        mapping = {
            (SourceType.YARD, DestinationType.YARD): MoveType.YARD_TO_YARD,
            (SourceType.YARD, DestinationType.TRAIN): MoveType.YARD_TO_TRAIN,
            (SourceType.YARD, DestinationType.TRUCK): MoveType.YARD_TO_TRUCK,
            (SourceType.TRAIN, DestinationType.YARD): MoveType.TRAIN_TO_YARD,
            (SourceType.TRAIN, DestinationType.TRUCK): MoveType.TRAIN_TO_TRUCK,
            (SourceType.TRUCK, DestinationType.YARD): MoveType.TRUCK_TO_YARD,
            (SourceType.TRUCK, DestinationType.TRAIN): MoveType.TRUCK_TO_TRAIN,
        }
        return mapping.get((source, dest), MoveType.YARD_TO_YARD)
    
    def _build_move_args(
        self,
        cont: MoveableContainer,
        dest: Destination
    ) -> Dict[str, Any]:
        """Build move args dictionary."""
        args = {"container_id": cont.container_id}
        
        # Add source reference
        if cont.source_type == SourceType.TRAIN:
            args["train_id"] = cont.source_id
        elif cont.source_type == SourceType.TRUCK:
            args["truck_id"] = cont.source_id
        
        # Add destination reference
        if dest.dest_type == DestinationType.YARD:
            args["placement"] = PlacementResult(
                row=dest.row,
                bay=dest.bay,
                tier=dest.tier,
                start_split=dest.start_split
            )
        elif dest.dest_type == DestinationType.TRAIN:
            args["train_id"] = dest.dest_id
        elif dest.dest_type == DestinationType.TRUCK:
            args["truck_id"] = dest.dest_id
        
        return args
    
    def _create_move(self, move_type: MoveType, args: Dict) -> Any:
        """Create Move object compatible with TLM."""
        from simulation.operations.terminal_manager import Move
        return Move(type=move_type, args=args)
    
    def _advance_time(self):
        """Advance simulation time by step_minutes."""
        self.env.current_time += timedelta(minutes=self.env.step_minutes)
    
    def _check_day_end(self) -> bool:
        """Check if day has ended."""
        if self.env.day_plan is None:
            return True
        day_end = self.env.day_plan.date.replace(hour=23, minute=59, second=0)
        return self.env.current_time >= day_end
    
    def step_all_cranes(
        self,
        agent: HierarchicalDQNAgent
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:
        """
        Execute hierarchical steps for all available cranes sequentially.
        
        Returns:
            Tuple of (next_state, total_reward, done, info)
        """
        total_reward = 0.0
        all_info: Dict[str, Any] = {
            "crane_results": [],
            "executed": [],
            "transitions": []
        }
        done = False
        
        # Handle arrivals/departures first
        self._handle_arrivals_departures()
        
        # Get idle cranes
        now = self.env.current_time
        idle_cranes = [
            c for c in self.env.cranes
            if c.busy_until is None or c.busy_until <= now
        ]
        
        if not idle_cranes:
            # All cranes busy - advance to next available
            next_time = min(
                c.busy_until for c in self.env.cranes 
                if c.busy_until is not None
            )
            self.env.current_time = next_time
            
            state = self.env.encoder.encode_with_forecast(
                self.env.trains, self.env.trucks, self.env.terminal_trucks,
                self.env.day_plan, self.env.current_time
            )
            done = self._check_day_end()
            return state, 0.0, done, all_info
        
        # Process each idle crane
        for crane in idle_cranes:
            result = self.step_hierarchical(agent, crane_id=crane.id)
            
            total_reward += result.reward
            all_info["crane_results"].append({
                "crane_id": crane.id,
                "reward": result.reward,
                "was_valid": result.was_valid_move
            })
            all_info["executed"].extend(result.info.get("executed", []))
            
            if result.transition is not None:
                all_info["transitions"].append(result.transition)
                agent.remember(result.transition)
            
            if result.done:
                done = True
                break
        
        # Get final state
        final_state = self.env.encoder.encode_with_forecast(
            self.env.trains, self.env.trucks, self.env.terminal_trucks,
            self.env.day_plan, self.env.current_time
        )
        
        return final_state, total_reward, done, all_info
    
    def _handle_arrivals_departures(self):
        """Handle train/truck arrivals and departures."""
        # This delegates to the underlying env's logic
        if hasattr(self.env, '_admit_arrivals_and_departures'):
            self.env._admit_arrivals_and_departures()
        
        # Update train heat
        self._update_train_heat()
