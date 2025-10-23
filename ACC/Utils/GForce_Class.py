class GForceCalculator:
    """
    This class calculates the G-force of a vehicle over time while it accelerates in CARLA.
    """

    def __init__(self,time_delta):
        # Store speeds (in m/s) as the vehicle moves
        self.speed_history = []
        # Store calculated g-force values
        self.g_force_values = []
        # Store last timestamp for time delta
        self.last_timestamp = None
        self.time_delta = time_delta

    def update_speed(self, current_speed):
        """
        Add the current speed (in m/s) to the list and calculate G-force if possible.
        :param current_speed: The vehicle's current speed in m/s
        :param current_time: The current CARLA simulation time in seconds
        """

        # If car is stopped, reset everything
        if current_speed < 0.1:
            self.speed_history.clear()
            self.g_force_values.clear()
            return

        # Record current speed
        self.speed_history.append(current_speed)

        # If we have enough speeds, start processing
        if len(self.speed_history) >= 5:
            # Split the list into 5 pairs (initial, final)
            portion_size = len(self.speed_history) // 2

            for i in range(portion_size):
                init_speed = self.speed_history[i]
                final_speed = self.speed_history[i + portion_size]

                # Calculate Δspeed
                delta_v = final_speed - init_speed

                # Calculate Δtime (negative init time delta as per instruction)
                delta_t = self.time_delta

                if delta_t == 0:
                    continue

                # Calculate acceleration
                acceleration = delta_v / delta_t

                # Calculate G-force
                g_force = acceleration / 9.81

                # Store the G-force value
                self.g_force_values.append(g_force)

            # Keep only the last few speeds to keep processing efficiently
            self.speed_history = self.speed_history[-portion_size:]


    def get_g_forces(self):
        """
        Returns the list of all calculated G-force values.
        """
        return self.g_force_values


if __name__ == '__main__':
    # Example of how this class could be used
    calculator = GForceCalculator(1)

    # Simulating speed readings over time
    speeds = [0, 5, 10, 15, 20, 31, 30, 35, 40, 45]  # in m/s
    #times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # in seconds

    for speed in speeds:
        calculator.update_speed(speed)

    print("Calculated G-forces over time:")
    print(calculator.get_g_forces())

