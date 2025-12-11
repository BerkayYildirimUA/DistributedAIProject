from app.data_processors.metrics_logger import iter_metrics
from app.data_processors.metrics_logger import clear_metrics_file
from app.data_processors.statistics_calculator import StatisticsCalculator
import app.constants as constants

def main_loop():
    # Create plots
    stats = StatisticsCalculator(reader_fn=iter_metrics)
    object_stats = StatisticsCalculator(reader_fn=iter_metrics)

    stats.add_metric_from_files(
        name="lead_distance_m",
        actual_file=constants.GT_LEAD_DISTANCE_FILE,
        estimated_file=constants.LEAD_DISTANCE_FILE,
        actual_key="lead_distance",
        estimated_key="lead_distance",
        frame_key="frame_id",
        plot_title="Predicted distance vs actual distance to vehicle in front",
        display_name="Lead distance (m)"
    )

    stats.add_metric_from_files(
        name="following distance vs safe following distance",
        actual_file=constants.GT_SAFE_FOLLOWING_DISTANCE_FILE,
        estimated_file=constants.LEAD_DISTANCE_FILE,
        actual_key="safe_following_distance",
        estimated_key="lead_distance",
        frame_key="frame_id",
        plot_title="Distance to vehicle in front vs the advised safe following distance",
        display_name="Following distance (m)"
    )

    stats.add_metric_from_files(
        name="Speed vs speed limit (ms)",
        actual_file=constants.GT_SPEED_LIMIT_FILE,
        estimated_file=constants.SPEED_FILE,
        actual_key="speed_limit",
        estimated_key="speed",
        frame_key="frame_id",
        plot_title="Driving speed vehicle vs the speed limit",
        display_name="Speed limit (m/s)"

    )
    stats.add_metric_from_files(
        name="g-force",
        actual_file=constants.G_FORCE_FILE,
        estimated_file=constants.G_FORCE_FILE,
        actual_key="force",
        estimated_key="force",
        frame_key="frame_id",
        plot_title="Experienced G-force vs conformable G-force",
        display_name="G-force"

    )

    object_stats.add_metric_from_files(
        name="objects_in_front_count",
        actual_file=constants.GT_OBJECTS_IN_FRONT_COUNT_FILE,
        estimated_file=constants.ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE,
        actual_key="ground_truth_objects",
        estimated_key="estimated_yolo_objects",
        frame_key="frame_id",
        plot_title="Ground-truth object count vs detected object count",
        display_name="Count"
    )

    object_stats.add_metric_from_files(
        name="tl_in_front_count",
        actual_file=constants.GT_TRAFFIC_LIGHT_COUNT_FILE,
        estimated_file=constants.ESTIMATED_TRAFFIC_LIGHT_COUNT_FILE,
        actual_key="ground_truth_traffic_lights",
        estimated_key="estimated_traffic_lights",
        frame_key="frame_id",
        plot_title="Ground-truth traffic light count vs detected traffic light count",
    )

    object_stats.add_metric_from_files(
        name="ts_in_front_count",
        actual_file=constants.GT_TRAFFIC_SIGN_COUNT_FILE,
        estimated_file=constants.ESTIMATED_TRAFFIC_SIGN_COUNT_FILE,
        actual_key="ground_truth_traffic_signs",
        estimated_key="estimated_traffic_signs",
        frame_key="frame_id",
        plot_title="Ground-truth traffic sign count vs detected traffic sign count",
    )

    object_stats.add_metric_from_files(
        name="vehicles_in_front_count",
        actual_file=constants.GT_VEHICLE_COUNT_FILE,
        estimated_file=constants.ESTIMATED_VEHICLE_COUNT_FILE,
        actual_key="ground_truth_vehicles_front_count",
        estimated_key="estimated_vehicles_front_count",
        frame_key="frame_id",
        plot_title="Ground-truth vehicle count vs detected vehicle count",
    )

    object_stats.add_metric_from_files(
        name="pedestrians_in_front_count",
        actual_file=constants.GT_PEDESTRIAN_COUNT_FILE,
        estimated_file=constants.ESTIMATED_PEDESTRIAN_COUNT_FILE,
        actual_key="ground_truth_pedestrians",
        estimated_key="estimated_pedestrians",
        frame_key="frame_id",
        plot_title="Ground-truth pedestrian count vs detected pedestrian count",
    )

    stats.plot_all("run_002_metrics.png", suptitle="Run 002 metrics")
    object_stats.plot_all("object_count_metrics.png", suptitle="Ground-truth vs Yolo objects")

    clear_metrics_file(constants.GT_LEAD_DISTANCE_FILE)
    clear_metrics_file(constants.LEAD_DISTANCE_FILE)
    clear_metrics_file(constants.GT_SAFE_FOLLOWING_DISTANCE_FILE)
    clear_metrics_file(constants.SPEED_FILE)
    clear_metrics_file(constants.GT_SPEED_LIMIT_FILE)
    clear_metrics_file(constants.G_FORCE_FILE)
    clear_metrics_file(constants.ESTIMATED_VEHICLE_DISTANCE_IN_FRONT_FILE)
    clear_metrics_file(constants.ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE)
    clear_metrics_file(constants.ESTIMATED_TRAFFIC_LIGHT_COUNT_FILE)
    clear_metrics_file(constants.GT_TRAFFIC_LIGHT_COUNT_FILE)
    clear_metrics_file(constants.ESTIMATED_TRAFFIC_SIGN_COUNT_FILE)
    clear_metrics_file(constants.GT_TRAFFIC_SIGN_COUNT_FILE)
    clear_metrics_file(constants.ESTIMATED_VEHICLE_COUNT_FILE)
    clear_metrics_file(constants.GT_VEHICLE_COUNT_FILE)
    clear_metrics_file(constants.ESTIMATED_PEDESTRIAN_COUNT_FILE)
    clear_metrics_file(constants.GT_PEDESTRIAN_COUNT_FILE)

    print("GETTING STATS")


if __name__ == '__main__':
    main_loop()
    print("Finished")