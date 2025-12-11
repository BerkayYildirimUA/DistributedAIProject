from app.data_processors.metrics_logger import iter_metrics
from app.data_processors.metrics_logger import clear_metrics_file
from app.data_processors.statistics_calculator import StatisticsCalculator
import app.constants as constants

def main_loop():
    # Create plots
    stats = StatisticsCalculator(reader_fn=iter_metrics)

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

    stats.add_metric_from_files(
        name="objects_in_front_count",
        actual_file=constants.ACTUAL_OBJECTS_IN_FRONT_COUNT_FILE,
        estimated_file=constants.ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE,
        actual_key="ground_truth_objects",
        estimated_key="estimated_yolo_objects",
        frame_key="frame_id",
        plot_title="Number of objects vs detected objects in front of the vehicle",
        display_name="Count"
    )

    stats.plot_all("run_002_metrics.png", suptitle="Run 002 metrics")

    clear_metrics_file(constants.GT_LEAD_DISTANCE_FILE)
    clear_metrics_file(constants.LEAD_DISTANCE_FILE)
    clear_metrics_file(constants.GT_SAFE_FOLLOWING_DISTANCE_FILE)

    clear_metrics_file(constants.SPEED_FILE)
    clear_metrics_file(constants.GT_SPEED_LIMIT_FILE)
    clear_metrics_file(constants.G_FORCE_FILE)

    clear_metrics_file(constants.ESTIMATED_VEHICLE_DISTANCE_IN_FRONT_FILE)
    clear_metrics_file(constants.ESTIMATED_OBJECT_IN_FRONT_COUNT_FILE)

    print("GETTING STATS")


if __name__ == '__main__':
    main_loop()
    print("Finished")