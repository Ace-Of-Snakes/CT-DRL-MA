# multihead_dqn/test_agent.py
"""Test script for Multi-Head DQN agent."""
import numpy as np
import torch

from simulation.rl.multihead_dqn.config import (
    MultiHeadDQNConfig, YardDims, BackboneConfig, HeadConfig, DQNConfig,
    ActionType, DestinationType
)
from simulation.rl.multihead_dqn.networks import MultiHeadQNetwork
from simulation.rl.multihead_dqn.agent import MultiHeadDQNAgent, ActionResult
from simulation.rl.multihead_dqn.replay_buffer import Transition, ReplayBuffer
from simulation.rl.multihead_dqn.state_utils import SplitLevelEncoder, ChannelSpec


def test_config():
    """Test configuration classes."""
    print("Testing configuration...")
    
    yard = YardDims(
        n_rows=5,
        n_splits=580,  # 58 bays * 10 split_factor
        n_tiers=5,
        n_bays=58,
        split_factor=10
    )
    
    cfg = MultiHeadDQNConfig(
        yard=yard,
        backbone=BackboneConfig(in_channels=8),
        heads=HeadConfig(),
        training=DQNConfig()
    )
    
    print(f"  Yard dims: {yard.spatial_shape}")
    print(f"  Total positions: {yard.total_positions}")
    print(f"  Device: {cfg.device}")
    print("Config OK")
    return cfg


def test_network(cfg: MultiHeadDQNConfig):
    """Test network forward passes."""
    print("\nTesting network...")
    
    device = torch.device(cfg.device)
    net = MultiHeadQNetwork(cfg.yard, cfg.backbone, cfg.heads).to(device)
    
    # Test shapes
    B = 2
    C = cfg.backbone.in_channels
    R, S, T = cfg.yard.spatial_shape
    
    # Random state
    state = torch.randn(B, C, R, S, T, device=device)
    
    # Encode
    feat_map, global_feat = net.encode_state(state)
    print(f"  Feature map: {feat_map.shape}")
    print(f"  Global feat: {global_feat.shape}")
    
    # Test action type head
    q_type = net.q_action_type(global_feat)
    print(f"  Q action type: {q_type.shape}")
    
    # Test container selection head
    occupancy = torch.rand(B, R, S, T, device=device) > 0.8  # ~20% occupied
    q_cont = net.q_container_selection(feat_map, occupancy)
    print(f"  Q container: {q_cont.shape}")
    
    # Test dest type head
    pos = torch.tensor([[0, 100, 0], [1, 200, 1]], device=device)
    cont_feat = net.extract_container_features(feat_map, pos)
    print(f"  Container feat: {cont_feat.shape}")
    
    q_dest = net.q_dest_type(global_feat, cont_feat)
    print(f"  Q dest type: {q_dest.shape}")
    
    # Test placement head
    validity = torch.rand(B, R, S, T, device=device) > 0.5
    q_place = net.q_placement(feat_map, global_feat, pos, validity)
    print(f"  Q placement: {q_place.shape}")
    
    # Test vehicle head
    V = 10
    vehicle_feats = torch.randn(B, V, cfg.heads.vehicle_feat_dim, device=device)
    vehicle_mask = torch.rand(B, V, device=device) > 0.3
    q_veh = net.q_vehicle(global_feat, cont_feat, vehicle_feats, vehicle_mask)
    print(f"  Q vehicle: {q_veh.shape}")
    
    # Test parking head
    P = 5
    parking_feats = torch.randn(B, P, cfg.heads.vehicle_feat_dim, device=device)
    parking_mask = torch.rand(B, P, device=device) > 0.5
    q_park = net.q_parking(global_feat, parking_feats, parking_mask)
    print(f"  Q parking: {q_park.shape}")
    
    print("Network OK")
    return net


def test_agent(cfg: MultiHeadDQNConfig):
    """Test agent action selection."""
    print("\nTesting agent...")
    
    agent = MultiHeadDQNAgent(cfg)
    
    C = cfg.backbone.in_channels
    R, S, T = cfg.yard.spatial_shape
    
    # Create test state
    state = np.random.randn(C, R, S, T).astype(np.float32)
    occupancy_mask = np.random.rand(R, S, T) > 0.85  # ~15% occupied
    validity_mask = np.random.rand(R, S, T) > 0.5
    
    # Vehicles
    V = 8
    vehicle_feats = np.random.randn(V, cfg.heads.vehicle_feat_dim).astype(np.float32)
    vehicle_mask = np.random.rand(V) > 0.4
    
    # Parking
    P = 3
    parking_feats = np.random.randn(P, cfg.heads.vehicle_feat_dim).astype(np.float32)
    parking_mask = np.random.rand(P) > 0.3
    
    # Test action selection
    for i in range(5):
        result = agent.act(
            state=state,
            occupancy_mask=occupancy_mask,
            validity_mask=validity_mask,
            vehicle_feats=vehicle_feats,
            vehicle_mask=vehicle_mask,
            parking_feats=parking_feats,
            parking_mask=parking_mask,
            epsilon=0.5  # 50% random for testing
        )
        
        print(f"  Action {i+1}: type={result.action_type.name}", end="")
        if result.action_type == ActionType.MOVE_CONTAINER:
            print(f", container={result.container_pos}", end="")
            if result.dest_type is not None:
                print(f", dest={result.dest_type.name}", end="")
                if result.dest_type == DestinationType.YARD:
                    print(f", placement={result.placement_pos}", end="")
                else:
                    print(f", vehicle={result.vehicle_idx}", end="")
        else:
            print(f", parking_idx={result.parking_idx}", end="")
        print()
    
    print("Agent OK")
    return agent


def test_replay(cfg: MultiHeadDQNConfig, agent: MultiHeadDQNAgent):
    """Test replay buffer and optimization."""
    print("\nTesting replay buffer and optimization...")
    
    C = cfg.backbone.in_channels
    R, S, T = cfg.yard.spatial_shape
    
    # Fill buffer with dummy transitions
    for i in range(100):
        transition = Transition(
            state=np.random.randn(C, R, S, T).astype(np.float32),
            action_type=ActionType.MOVE_CONTAINER,
            container_pos=(np.random.randint(R), np.random.randint(S), np.random.randint(T)),
            dest_type=DestinationType.YARD,
            placement_pos=(np.random.randint(R), np.random.randint(S), np.random.randint(T)),
            reward=np.random.randn(),
            next_state=np.random.randn(C, R, S, T).astype(np.float32),
            done=i % 20 == 0
        )
        agent.remember(transition)
    
    print(f"  Buffer size: {len(agent.replay)}")
    
    # Test optimization
    losses = []
    for i in range(10):
        loss = agent.optimize()
        losses.append(loss)
    
    print(f"  Losses: {[f'{l:.4f}' for l in losses[:5]]}")
    print("Replay/Optimization OK")


def test_state_encoder():
    """Test state encoding utilities."""
    print("\nTesting state encoder...")
    
    encoder = SplitLevelEncoder(
        n_rows=5,
        n_bays=58,
        n_tiers=5,
        split_factor=10
    )
    
    print(f"  Total splits: {encoder.total_splits}")
    print(f"  Channels: {ChannelSpec.num_channels()}")
    
    # Test validity mask computation
    R, S, T = 5, 580, 5
    occupancy = np.random.rand(R, S, T) > 0.9
    
    validity = encoder.get_validity_mask(
        occupancy_mask=occupancy,
        n_splits_needed=10  # 40ft container
    )
    
    print(f"  Occupancy shape: {occupancy.shape}")
    print(f"  Validity shape: {validity.shape}")
    print(f"  Occupied positions: {occupancy.sum()}")
    print(f"  Valid placements: {validity.sum()}")
    
    print("State encoder OK")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Multi-Head DQN Agent Tests")
    print("=" * 50)
    
    try:
        cfg = test_config()
        net = test_network(cfg)
        agent = test_agent(cfg)
        test_replay(cfg, agent)
        test_state_encoder()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗-- Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())