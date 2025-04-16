"""
This simple example to play around with animation of a cart pole using an LQR controller.

This is based on the bouncing_rectangle example from the Python arcade docs.
"""

import arcade
import numpy as np
from cart_pole_dynamics import CartPolePlant, LQRController

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Cart Pole Example"

# Cart info
CART_WIDTH = 100
CART_HEIGHT = 50
CART_COLOR = arcade.color.RED
CART_WHEELS_RADIUS = 10

# Pole info
POLE_COLOR = arcade.color.BLUE_SAPPHIRE

GROUND_LEVEL = 240

BACKGROUND_COLOR = arcade.csscolor.SKY_BLUE


class CartPole:
    """Cart Pole Visualization"""

    def __init__(
        self, initial_x_center_cart, initial_y_bottom_cart, pole_length, scale_factor
    ):
        self.center_x = initial_x_center_cart
        self.center_y = initial_y_bottom_cart + CART_WHEELS_RADIUS + CART_HEIGHT
        self.theta = 0
        self.length = pole_length * scale_factor

    def update(self, new_x, theta):
        self.center_x = new_x
        self.theta = theta

    def draw(self):
        # Cart
        arcade.draw_rect_filled(
            arcade.rect.XYWH(
                self.center_x, self.center_y - CART_HEIGHT / 2, CART_WIDTH, CART_HEIGHT
            ),
            CART_COLOR,
        )

        # Wheels
        arcade.draw_circle_filled(
            self.center_x + CART_WIDTH / 2,
            self.center_y - CART_HEIGHT,
            CART_WHEELS_RADIUS,
            arcade.color.BLACK,
        )
        arcade.draw_circle_filled(
            self.center_x - CART_WIDTH / 2,
            self.center_y - CART_HEIGHT,
            CART_WHEELS_RADIUS,
            arcade.color.BLACK,
        )

        # Pole
        end_x = self.center_x + 2 * self.length * np.sin(self.theta)
        end_y = self.center_y + 2 * self.length * np.cos(self.theta)
        arcade.draw_line(self.center_x, self.center_y, end_x, end_y, POLE_COLOR, 10)


class GameView(arcade.View):
    """Main application class which simulates the controller and plant"""

    def __init__(self):
        super().__init__()

        # LQR tuning and initial state
        Q = np.diag([1, 3, 10, 1])
        R = 0.1
        self.timestep = 0.01
        initial_state = np.array([0.0, 0.0, np.pi / 10.0, 0.0])

        # Create the plant and the controller
        pole_length = 0.3
        self.plant = CartPolePlant(
            length_value=pole_length,
            mass_cart_value=0.5,
            mass_pole_value=0.2,
            inertia_pole_value=0.006,
            initial_state=initial_state,
        )

        self.controller = LQRController(
            self.plant.get_A(),
            self.plant.get_B(),
            self.plant.get_C(),
            self.plant.get_D(),
            Q,
            R,
            self.timestep,
        )

        self.scaling_factor = 250
        self.starting_x = 200

        # Create the cart pole animation
        self.cart_pole = CartPole(
            self.starting_x, GROUND_LEVEL, pole_length, self.scaling_factor
        )

        # Set background color
        self.background_color = BACKGROUND_COLOR

    def on_update(self, delta_time):
        # Simulate the cart
        u = self.controller.get_control_input(self.plant.get_state())
        self.plant.step(u, self.timestep)
        state = self.plant.get_state()

        # Get the states and scale for the visualization
        x = state[0] * self.scaling_factor + self.starting_x
        theta = state[2]

        # Update the cart pole visual
        self.cart_pole.update(x, theta)

    def draw_background(self):
        # Background sun and lawn
        arcade.draw_lrbt_rectangle_filled(
            0, 600, 0, GROUND_LEVEL, arcade.csscolor.GREEN
        )
        arcade.draw_circle_filled(100, 500, 30, arcade.color.YELLOW)

    def on_draw(self):
        """Render the screen."""

        self.clear()
        self.draw_background()
        self.cart_pole.draw()

    def on_key_press(self, key, _modifiers):
        """Handle key presses, a space bar to end the visualization"""
        if key == arcade.key.SPACE:
            arcade.close_window()


def main():
    """Main function"""
    # Create a window class
    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)

    # Create the GameView
    game = GameView()

    # Show GameView on screen
    window.show_view(game)

    # Start the arcade game loop
    arcade.run()


if __name__ == "__main__":
    main()
