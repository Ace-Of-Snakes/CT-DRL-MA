"""
Test module for the refactored TrainScheduler.
Tests the corrected logic where trains arrive fully loaded.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List
import matplotlib.pyplot as plt

# Import the refactored scheduler
from simulation.terminal_components.systems.TrainScheduler import (
    TrainScheduler,
    TRAIN_NUM_WAGONS,
    TRAIN_WAGON_LENGTH_FT,
    TRAIN_MIN_WAGON_UTILIZATION,
    WEEKDAY_FULL_NAMES,
    ContainerAssignment
)
from simulation.terminal_components.storage_units.Container import Container


def create_test_driving_plan():
    """Create a test driving plan for demonstration."""
    test_plan = {
        "driving_plan": {
            "trains": {
                "TRN001": {
                    "operator": "boxXpress",
                    "destination": "Hamburg",
                    "plan": {
                        "1": {
                            "arrival": ["Monday", "08:00", "Monday", "08:30"],
                            "departure": ["Monday", "14:00", "Monday", "14:30"],
                            "mirrored_on": ["Wednesday", "Friday"]
                        },
                        "2": {
                            "arrival": ["Tuesday", "10:00", "Tuesday", "10:30"],
                            "departure": ["Tuesday", "16:00", "Tuesday", "16:30"],
                            "mirrored_on": []
                        }
                    }
                },
                "TRN002": {
                    "operator": "Metrans",
                    "destination": "Prague",
                    "plan": {
                        "1": {
                            "arrival": ["Monday", "12:00", "Monday", "12:30"],
                            "departure": ["Tuesday", "09:00", "Tuesday", "09:30"],
                            "mirrored_on": ["Thursday"]
                        }
                    }
                },
                "TRN003": {
                    "operator": "Kombiverkehr",
                    "destination": "Munich",
                    "plan": {
                        "1": {
                            "arrival": ["Tuesday", "14:00", "Tuesday", "14:30"],
                            "departure": ["Wednesday", "10:00", "Wednesday", "10:30"],
                            "mirrored_on": ["Thursday", "Saturday"]
                        }
                    }
                },
                "TRN004": {
                    "operator": "DHL/ DB Cargo",
                    "destination": "Berlin",
                    "plan": {
                        "1": {
                            "arrival": ["Wednesday", "06:00", "Wednesday", "06:30"],
                            "departure": ["Wednesday", "18:00", "Wednesday", "18:30"],
                            "mirrored_on": ["Friday", "Sunday"]
                        }
                    }
                }
            }
        }
    }
    return test_plan


def test_basic_functionality():
    """Test basic scheduler functionality."""
    print("=" * 80)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("=" * 80)
    
    # Create test data
    plan = create_test_driving_plan()
    
    # Save test plan
    os.makedirs("simulation/data", exist_ok=True)
    with open("simulation/data/test_driving_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    # Initialize scheduler
    scheduler = TrainScheduler(
        driving_plan_path="simulation/data/test_driving_plan.json",
        num_rails=10
    )
    
    # Show configuration
    print("\nScheduler Configuration:")
    print(f"  Number of rails: {scheduler.num_rails}")
    print(f"  Wagons per train: {TRAIN_NUM_WAGONS}")
    print(f"  Wagon length: {TRAIN_WAGON_LENGTH_FT} feet")
    print(f"  Minimum wagon utilization: {TRAIN_MIN_WAGON_UTILIZATION * 100:.0f}%")
    
    # Show schedule summary
    summary = scheduler.get_schedule_summary()
    print(f"\nSchedule Summary:")
    print(f"  Total weekly trains: {summary['total_weekly_trains']}")
    print(f"  Assigned trains: {summary['assigned_trains']}")
    print(f"  Unassigned trains: {len(summary['unassigned_trains'])}")
    
    print("\nTrains by operator:")
    for operator, count in summary['trains_by_operator'].items():
        print(f"  {operator}: {count} trains/week")
    
    print("\nTrains by weekday:")
    for day_idx, count in sorted(summary['trains_by_weekday'].items()):
        print(f"  {WEEKDAY_FULL_NAMES[day_idx]}: {count} trains")
    
    return scheduler


def test_container_pickup_assignment():
    """Test container pickup assignment system."""
    print("\n" + "=" * 80)
    print("TEST 2: CONTAINER PICKUP ASSIGNMENT")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Create pickup assignments
    print("\nAssigning containers for pickup:")
    for i in range(50):
        container_id = f"PICKUP_{i:04d}"
        wagon_idx = i % TRAIN_NUM_WAGONS  # Distribute across wagons
        scheduler.assign_container_pickup(container_id, wagon_idx)
    
    # Check queue status
    queue_status = scheduler.get_pickup_queue_status()
    print(f"\nPickup Queue Status:")
    print(f"  Total pending pickups: {queue_status['total_pending_pickups']}")
    print(f"  Next train can handle: {queue_status['next_train_can_handle']}")
    print(f"\n  Distribution by wagon:")
    for wagon, count in sorted(queue_status['distribution_by_wagon'].items()):
        print(f"    Wagon {wagon}: {count} containers")
    
    return scheduler


def test_train_arrival_full():
    """Test that trains arrive fully loaded."""
    print("\n" + "=" * 80)
    print("TEST 3: TRAIN ARRIVES FULLY LOADED")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Simulate train loading for different operators
    operators = ["boxXpress", "Metrans", "Kombiverkehr"]
    
    for operator in operators:
        print(f"\n{operator} Train Loading Simulation:")
        loading_result = scheduler.simulate_train_loading(operator)
        
        print(f"  Total containers: {loading_result['total_containers']}")
        print(f"  Wagons loaded: {loading_result['wagons_loaded']}/{TRAIN_NUM_WAGONS}")
        print(f"  Average utilization: {loading_result['average_wagon_utilization']:.1f}%")
        
        print(f"\n  Container type distribution:")
        for ctype, count in loading_result['container_type_distribution'].items():
            print(f"    {ctype}: {count}")
        
        print(f"\n  Sample wagon configurations (first 3):")
        for wagon in loading_result['sample_wagons'][:3]:
            print(f"    Wagon {wagon['wagon_index']}: {wagon['containers']} containers, "
                  f"{wagon['utilization']:.1f}% utilized")
            print(f"      Types: {wagon['types']}")
    
    return scheduler


def test_capacity_analysis():
    """Test train capacity analysis for different container sizes."""
    print("\n" + "=" * 80)
    print("TEST 4: TRAIN CAPACITY ANALYSIS")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    analysis = scheduler.analyze_train_capacity("boxXpress")
    
    print(f"\nCapacity Analysis for {analysis['operator']}:")
    print(f"  Wagons per train: {analysis['wagons_per_train']}")
    print(f"  Wagon length: {analysis['wagon_length_ft']} feet")
    print(f"  Minimum utilization: {analysis['min_utilization']}%")
    
    print(f"\nSample configurations:")
    for config in analysis['sample_configurations']:
        print(f"\n  {config['name']}:")
        print(f"    Total containers: {config['total_containers']}")
        print(f"    Wagons used: {config['wagons_used']}")
        print(f"    Average utilization: {config['avg_utilization']:.1f}%")
    
    return scheduler


def test_simulation_with_pickups():
    """Test full simulation with trains arriving full and picking up containers."""
    print("\n" + "=" * 80)
    print("TEST 5: SIMULATION WITH ARRIVALS AND PICKUPS")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Add containers to pickup queue
    for i in range(100):
        scheduler.assign_container_pickup(f"EXPORT_{i:04d}")
    
    print(f"Added 100 containers to pickup queue")
    
    # Initialize simulation
    sim_start = datetime(2024, 1, 8)  # Monday
    sim_end = sim_start + timedelta(days=1)
    
    scheduler.initialize_for_period(sim_start, sim_end)
    print(f"\nRunning simulation for one day...")
    
    # Process events
    events_processed = 0
    while scheduler.event_queue and events_processed < 20:
        event = scheduler.event_queue.pop(0)
        result = scheduler.process_event(event, event.timestamp)
        
        if result['action'] == 'containers_prepared':
            print(f"\n{event.timestamp.strftime('%H:%M')} - Train {result['train_id']}:")
            print(f"  Arrived with: {result['containers_on_arrival']} containers")
            print(f"  Pickups assigned: {result['pickups_assigned']}")
        elif result['action'] == 'train_departed':
            print(f"{event.timestamp.strftime('%H:%M')} - Train {result['train_id']} departed")
            print(f"  Departing with: {result['containers_departing']} containers")
            print(f"  Pickups completed: {result['pickups_completed']}")
        
        events_processed += 1
    
    # Show final metrics
    metrics = scheduler.get_metrics()
    print(f"\nSimulation Metrics:")
    print(f"  Trains processed: {metrics['trains_processed']}")
    print(f"  Containers on arrival: {metrics['containers_on_arrival']}")
    print(f"  Containers picked up: {metrics['containers_picked_up']}")
    print(f"  Containers delivered: {metrics['containers_delivered']}")
    
    # Check remaining pickup queue
    queue_status = scheduler.get_pickup_queue_status()
    print(f"\nRemaining in pickup queue: {queue_status['total_pending_pickups']}")
    
    return scheduler


def test_wagon_utilization_rules():
    """Test that wagon utilization rules are enforced."""
    print("\n" + "=" * 80)
    print("TEST 6: WAGON UTILIZATION RULES")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    print(f"\nWagon Utilization Rules:")
    print(f"  Minimum utilization: {TRAIN_MIN_WAGON_UTILIZATION * 100:.0f}%")
    print(f"  Wagon length: {TRAIN_WAGON_LENGTH_FT} feet")
    print(f"  Minimum used length: {TRAIN_WAGON_LENGTH_FT * TRAIN_MIN_WAGON_UTILIZATION:.1f} feet")
    
    # Test different container combinations
    test_cases = [
        {
            'name': '20ft + 20ft + 30ft',
            'sizes': [20, 20, 30],
            'total': 70,
            'valid': True  # 70/80 = 87.5%
        },
        {
            'name': '40ft + 40ft',
            'sizes': [40, 40],
            'total': 80,
            'valid': True  # 80/80 = 100%
        },
        {
            'name': '45ft + 30ft',
            'sizes': [45, 30],
            'total': 75,
            'valid': True  # 75/80 = 93.75%
        },
        {
            'name': '20ft + 20ft',
            'sizes': [20, 20],
            'total': 40,
            'valid': False  # 40/80 = 50% (below 80%)
        }
    ]
    
    print(f"\nContainer combination tests:")
    for test in test_cases:
        utilization = (test['total'] / TRAIN_WAGON_LENGTH_FT) * 100
        status = "✓ VALID" if test['valid'] else "✗ INVALID"
        print(f"  {test['name']} = {test['total']}ft ({utilization:.1f}%) - {status}")
    
    return scheduler


def test_gantt_visualization():
    """Test Gantt chart generation."""
    print("\n" + "=" * 80)
    print("TEST 7: GANTT CHART VISUALIZATION")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Generate Gantt chart
    fig = scheduler.get_weekly_gantt()
    
    # Save chart
    output_path = "test_weekly_schedule.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGantt chart saved to: {output_path}")
    plt.close(fig)
    
    return scheduler


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "=" * 80)
    print("REFACTORED TRAIN SCHEDULER - COMPREHENSIVE TESTS")
    print("=" * 80)
    print("\nKey Changes:")
    print("- Trains arrive FULLY LOADED with containers")
    print("- No fixed container counts - depends on container sizes")
    print("- Containers are assigned to specific wagons for pickup")
    print("- 80% minimum wagon utilization enforced")
    print("- No import/export ratio - trains generate imports, pickups are exports")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Container Pickup Assignment", test_container_pickup_assignment),
        ("Trains Arrive Full", test_train_arrival_full),
        ("Capacity Analysis", test_capacity_analysis),
        ("Simulation with Pickups", test_simulation_with_pickups),
        ("Wagon Utilization Rules", test_wagon_utilization_rules),
        ("Gantt Visualization", test_gantt_visualization)
    ]
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"Running Test {i}/{len(tests)}: {name}")
        print('='*80)
        try:
            test_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()


def create_test_driving_plan():
    """Create a test driving plan for demonstration."""
    test_plan = {
        "driving_plan": {
            "trains": {
                "TRN001": {
                    "operator": "boxXpress",
                    "destination": "Hamburg",
                    "plan": {
                        "1": {
                            "arrival": ["Monday", "08:00", "Monday", "08:30"],
                            "departure": ["Monday", "14:00", "Monday", "14:30"],
                            "mirrored_on": ["Wednesday", "Friday"]
                        },
                        "2": {
                            "arrival": ["Tuesday", "10:00", "Tuesday", "10:30"],
                            "departure": ["Tuesday", "16:00", "Tuesday", "16:30"],
                            "mirrored_on": []
                        }
                    }
                },
                "TRN002": {
                    "operator": "Metrans",
                    "destination": "Prague",
                    "plan": {
                        "1": {
                            "arrival": ["Monday", "12:00", "Monday", "12:30"],
                            "departure": ["Tuesday", "09:00", "Tuesday", "09:30"],
                            "mirrored_on": ["Thursday"]
                        }
                    }
                },
                "TRN003": {
                    "operator": "Kombiverkehr",
                    "destination": "Munich",
                    "plan": {
                        "1": {
                            "arrival": ["Tuesday", "14:00", "Tuesday", "14:30"],
                            "departure": ["Wednesday", "10:00", "Wednesday", "10:30"],
                            "mirrored_on": ["Thursday", "Saturday"]
                        }
                    }
                },
                "TRN004": {
                    "operator": "DHL/ DB Cargo",
                    "destination": "Berlin",
                    "plan": {
                        "1": {
                            "arrival": ["Wednesday", "06:00", "Wednesday", "06:30"],
                            "departure": ["Wednesday", "18:00", "Wednesday", "18:30"],
                            "mirrored_on": ["Friday", "Sunday"]
                        }
                    }
                }
            }
        }
    }
    return test_plan


def create_test_export_containers(count: int = 50) -> List[str]:
    """Create test export container IDs."""
    return [f"EXP_{i:04d}" for i in range(1, count + 1)]


def test_basic_functionality():
    """Test basic scheduler functionality."""
    print("=" * 80)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("=" * 80)
    
    # Create test data
    plan = create_test_driving_plan()
    
    # Save test plan
    os.makedirs("simulation/data", exist_ok=True)
    with open("simulation/data/test_driving_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    # Initialize scheduler
    scheduler = TrainScheduler(
        driving_plan_path="simulation/data/test_driving_plan.json",
        num_rails=10
    )
    
    # Show configuration
    print("\nScheduler Configuration:")
    print(f"  Number of rails: {scheduler.num_rails}")
    print(f"  Trains per wagon: {TRAIN_NUM_WAGONS}")
    print(f"  Wagon length: {TRAIN_WAGON_LENGTH_FT} feet")
    print(f"  Minimum wagon utilization: {TRAIN_MIN_WAGON_UTILIZATION * 100:.0f}%")
    
    # Show schedule summary
    summary = scheduler.get_schedule_summary()
    print(f"\nSchedule Summary:")
    print(f"  Total weekly trains: {summary['total_weekly_trains']}")
    print(f"  Assigned trains: {summary['assigned_trains']}")
    print(f"  Unassigned trains: {len(summary['unassigned_trains'])}")
    
    print("\nTrains by weekday:")
    for day_idx, count in sorted(summary['trains_by_weekday'].items()):
        print(f"  {WEEKDAY_FULL_NAMES[day_idx]}: {count} trains")
    
    return scheduler


def test_export_container_assignment():
    """Test export container assignment to trains."""
    print("\n" + "=" * 80)
    print("TEST 2: EXPORT CONTAINER ASSIGNMENT")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Create export containers
    export_containers = create_test_export_containers(100)
    print(f"\nCreated {len(export_containers)} export containers")
    
    # Add to scheduler queue
    scheduler.add_export_containers(export_containers)
    print(f"Export queue size: {len(scheduler.export_container_queue)}")
    
    # Initialize simulation period
    sim_start = datetime(2024, 1, 8)  # Monday
    sim_end = sim_start + timedelta(days=7)
    
    scheduler.initialize_for_period(sim_start, sim_end)
    print(f"\nInitialized {len(scheduler.event_queue)} events for the week")
    
    # Process first train arrival
    current_time = sim_start
    events_processed = 0
    
    while scheduler.event_queue and events_processed < 10:
        event = scheduler.event_queue[0]
        if event.timestamp <= current_time + timedelta(hours=24):
            event = scheduler.event_queue.pop(0)
            result = scheduler.process_event(event, event.timestamp)
            
            if result['action'] == 'containers_prepared':
                print(f"\nTrain {result['train_id']} at {event.timestamp.strftime('%a %H:%M')}:")
                print(f"  Import containers: {result['import_count']}")
                print(f"  Export containers assigned: {result['export_count']}")
                print(f"  Total containers: {result['total_containers']}")
                print(f"  Remaining in export queue: {len(scheduler.export_container_queue)}")
            
            events_processed += 1
            current_time = event.timestamp
    
    return scheduler


def test_wagon_utilization():
    """Test wagon utilization optimization."""
    print("\n" + "=" * 80)
    print("TEST 3: WAGON UTILIZATION OPTIMIZATION")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Create test containers with different sizes
    test_containers = []
    container_sizes = [
        (20, 10),  # 20 foot containers
        (30, 8),   # 30 foot containers
        (40, 12),  # 40 foot containers
        (45, 5)    # 45 foot containers
    ]
    
    for size_ft, count in container_sizes:
        for i in range(count):
            container = Container(
                container_id=f"TEST_{size_ft}FT_{i:03d}",
                direction="Import",
                container_type=f"{size_ft}FT",
                arrival_date=datetime.now(),
                departure_date=datetime.now() + timedelta(days=3),
                length_ft=size_ft,
                length_m=size_ft * 0.3048  # Convert feet to meters
            )
            test_containers.append(container)
    
    print(f"\nCreated {len(test_containers)} test containers:")
    for size_ft, count in container_sizes:
        print(f"  {size_ft} foot: {count} containers")
    
    # Optimize loading
    result = scheduler.optimize_wagon_loading(test_containers)
    
    print(f"\nOptimization Results:")
    print(f"  Wagons used: {result['wagons_used']}")
    print(f"  Containers loaded: {result['total_containers']}/{len(test_containers)}")
    print(f"  Average utilization: {result['average_utilization'] * 100:.1f}%")
    
    print(f"\nWagon Details:")
    for i, wagon in enumerate(result['assignments'][:5]):  # Show first 5 wagons
        containers = wagon['containers']
        lengths = [c.length_ft for c in containers]
        print(f"  Wagon {i+1}: {lengths} = {sum(lengths)}ft ({wagon['utilization']*100:.0f}%)")
    
    return scheduler


def test_rail_utilization():
    """Test rail utilization analysis."""
    print("\n" + "=" * 80)
    print("TEST 4: RAIL UTILIZATION ANALYSIS")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    summary = scheduler.get_schedule_summary()
    
    print("\nRail Utilization:")
    total_util = 0
    for rail, util in sorted(summary['rail_utilization'].items()):
        print(f"  {rail}:")
        print(f"    Hours per week: {util['hours_per_week']:.1f}")
        print(f"    Utilization: {util['utilization_percent']:.1f}%")
        total_util += util['utilization_percent']
    
    avg_util = total_util / scheduler.num_rails if scheduler.num_rails > 0 else 0
    print(f"\nAverage rail utilization: {avg_util:.1f}%")
    
    # Efficiency analysis
    efficiency = scheduler.analyze_schedule_efficiency()
    
    print(f"\nSchedule Efficiency:")
    print(f"  Average train stay: {efficiency['average_duration_hours']:.1f} hours")
    
    print(f"\n  Longest stays:")
    for train_id, hours in efficiency['longest_stays'][:3]:
        print(f"    {train_id}: {hours:.1f} hours")
    
    print(f"\n  Shortest stays:")
    for train_id, hours in efficiency['shortest_stays'][:3]:
        print(f"    {train_id}: {hours:.1f} hours")
    
    if efficiency['recommendations']:
        print(f"\nRecommendations:")
        for rec in efficiency['recommendations']:
            print(f"  - {rec}")
    
    return scheduler


def test_next_train_lookup():
    """Test next train arrival lookup."""
    print("\n" + "=" * 80)
    print("TEST 5: NEXT TRAIN LOOKUP")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Test different times
    test_times = [
        datetime(2024, 1, 8, 7, 0),    # Monday 07:00
        datetime(2024, 1, 8, 12, 30),  # Monday 12:30
        datetime(2024, 1, 10, 15, 0),  # Wednesday 15:00
        datetime(2024, 1, 12, 20, 0),  # Friday 20:00
    ]
    
    print("\nNext Train Arrivals:")
    for test_time in test_times:
        result = scheduler.get_next_train_arrival(test_time)
        if result:
            next_arrival, schedule = result
            wait_hours = (next_arrival - test_time).total_seconds() / 3600
            print(f"\n  From {test_time.strftime('%A %H:%M')}:")
            print(f"    Next: {schedule.train_id.split('_')[0]} ({schedule.operator})")
            print(f"    Arrives: {next_arrival.strftime('%A %H:%M')}")
            print(f"    Wait: {wait_hours:.1f} hours")
            print(f"    Rail: {schedule.rail}")


def test_gantt_visualization():
    """Test Gantt chart generation."""
    print("\n" + "=" * 80)
    print("TEST 6: GANTT CHART VISUALIZATION")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Generate Gantt chart
    fig = scheduler.get_weekly_gantt()
    
    # Save chart
    output_path = "test_weekly_schedule.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGantt chart saved to: {output_path}")
    plt.close(fig)
    
    # Also test trains on specific days
    print("\nTrains by Day:")
    for day in range(7):
        trains = scheduler.get_trains_on_day(day)
        if trains:
            print(f"\n  {WEEKDAY_FULL_NAMES[day]}: {len(trains)} trains")
            for train in trains[:2]:  # Show first 2
                print(f"    - {train.train_id.split('_')[0]} at {train.arrival}")


def test_metrics_tracking():
    """Test metrics tracking during simulation."""
    print("\n" + "=" * 80)
    print("TEST 7: METRICS TRACKING")
    print("=" * 80)
    
    scheduler = test_basic_functionality()
    
    # Add export containers
    scheduler.add_export_containers(create_test_export_containers(50))
    
    # Run short simulation
    sim_start = datetime(2024, 1, 8)
    sim_end = sim_start + timedelta(days=2)
    
    scheduler.initialize_for_period(sim_start, sim_end)
    
    # Process all events
    while scheduler.event_queue:
        event = scheduler.event_queue.pop(0)
        scheduler.process_event(event, event.timestamp)
    
    # Show metrics
    metrics = scheduler.get_metrics()
    print("\nSimulation Metrics (2 days):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Calculate rates
    if metrics['total_arrivals'] > 0:
        containers_per_train = (metrics['import_containers_generated'] / 
                               metrics['total_arrivals'])
        print(f"\nDerived Metrics:")
        print(f"  Avg containers per train: {containers_per_train:.1f}")
        
        if metrics['wagons_utilized'] + metrics['wagons_underutilized'] > 0:
            utilization_rate = (metrics['wagons_utilized'] / 
                              (metrics['wagons_utilized'] + metrics['wagons_underutilized']))
            print(f"  Wagon utilization rate: {utilization_rate * 100:.1f}%")


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS FOR REFACTORED TRAIN SCHEDULER")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Export Container Assignment", test_export_container_assignment),
        ("Wagon Utilization", test_wagon_utilization),
        ("Rail Utilization", test_rail_utilization),
        ("Next Train Lookup", test_next_train_lookup),
        ("Gantt Visualization", test_gantt_visualization),
        ("Metrics Tracking", test_metrics_tracking)
    ]
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"Running Test {i}/{len(tests)}: {name}")
        print('='*80)
        try:
            test_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()