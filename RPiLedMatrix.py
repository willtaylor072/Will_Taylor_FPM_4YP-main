# RPiLedMatrix Class
# Author: Álvaro Fernández Galiana
# Email: alvaro.fernandezgaliana@gmail.com
#
# Copyright (c) 2024 Álvaro Fernández Galiana
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import time
import spidev
import numpy as np

class RPiLedMatrix:
    """
    A class to control a 16x16 LED matrix using the Unicorn HAT HD on a Raspberry Pi.
    
    Methods:
        __init__(self, rotation=0)
        set_brightness(self, b)
        set_rotation(self, rot)
        off(self)
        close(self)
        on_all(self)
        on_all_color(self, color)
        set_pixel(self, x, y, color, clear=True)
        show_pixel(self, x, y, color, clear=True, brightness=None)
        set_circle(self, radius, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, filled=True)
        show_circle(self, radius, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, brightness=None, filled=True)
        set_square(self, side, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, filled=True)
        show_square(self, side, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, brightness=None, filled=True)
        set_cross(self, thickness, span, color='white', offset_x=0, offset_y=0, clear=True)
        show_cross(self, thickness, span, color='white', offset_x=0, offset_y=0, clear=True, brightness=None)
        set_half(self, half, color, clear=True)
        show_half(self, half, color, clear=True, brightness=None)
        clear(self)
        set_all(self, r, g, b)
        show(self, brightness=None)
        _get_color(self, color)
    """
    
    def __init__(self, rotation=0):
        """
        Initialize the LED matrix with given rotation.

        Parameters:
            rotation (int): The rotation of the display (0, 90, 180, 270).
        """
        self.__version__ = '0.0.3'

        self._spi = spidev.SpiDev()
        self._spi.open(0, 0)
        self._spi.max_speed_hz = 9000000
        self._SOF = 0x72
        self._DELAY = 1.0 / 120

        self.WIDTH = 16
        self.HEIGHT = 16

        self._rotation = rotation
        self._brightness = 0.5
        self._buf = np.zeros((self.WIDTH, self.HEIGHT, 3), dtype=int)

        # Predefined colors
        self.colors = {
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'black': (0, 0, 0),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            'gray': (128, 128, 128),
            'lime': (0, 255, 0),
            'indigo': (75, 0, 130),
            'violet': (238, 130, 238)
        }


    def set_brightness(self, b):
        """
        Set the display brightness between 0.0 and 1.0.

        Parameters:
            b (float): Brightness level (0.0 to 1.0).
        """
        self._brightness = b
        
    def set_rotation(self, rot):
        """
        Set the display rotation.

        Parameters:
            rot (int): The rotation of the display (0, 90, 180, 270).
        """
        self._rotation = rot

    def off(self):
        """
        Turn off all LEDs.
        """
        self.clear()
        self.show()

    def close(self):
        """
        Close the SPI device.
        """
        self._spi.close()

    def on_all(self, brightness=None):
        """
        Turn on all LEDs with the current brightness.
        """
        self.set_all(255, 255, 255)
        self.show(brightness)

    def on_all_color(self, color='white', brightness=None):
        """
        Turn on all LEDs with the specified color.

        Parameters:
            color (str or tuple): Color name or RGB tuple.
        """
        r, g, b = self._get_color(color)
        self.set_all(r, g, b)
        self.show(brightness)

    def set_pixel(self, x, y, color='white', clear=True):
        """
        Set a single pixel to an RGB color.

        Parameters:
            x (int): X-coordinate of the pixel.
            y (int): Y-coordinate of the pixel.
            color (str or tuple): Color name or RGB tuple. Defaul white.
            clear (bool): Whether to clear the buffer before setting the pixel.
        """
        if clear: self.clear()
        r, g, b = self._get_color(color)
        self._buf[x][y] = r, g, b
        
    def show_pixel(self, x, y, color='white', clear=True, brightness=None):
        """
        Set and show a single pixel to an RGB color.

        Parameters:
            x (int): X-coordinate of the pixel.
            y (int): Y-coordinate of the pixel.
            color (str or tuple): Color name or RGB tuple. Defaul white.
            clear (bool): Whether to clear the buffer before setting the pixel.
            brightness (float): Brightness level (0.0 to 1.0).
        """
        self.set_pixel(x, y, color, clear)
        self.show(brightness)
        
    
    def set_pixels(self, leds, color='white', clear=True):
        """
        Set a list of LEDs to the specified color.

        Parameters:
            leds (list of tuples): List of (x, y) tuples specifying LED positions.
            color (str or tuple): Color name or RGB tuple.
            clear (bool): Whether to clear the buffer before setting the LEDs.
        """
        if clear: self.clear()
        r, g, b = self._get_color(color)
        for x, y in leds:
            if 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT:
                self._buf[x][y] = [r, g, b]
                

    def show_pixels(self, leds, color='white', clear=True, brightness=None):
        """
        Set and show a list of LEDs with the specified color.

        Parameters:
            leds (list of tuples): List of (x, y) tuples specifying LED positions.
            color (str or tuple): Color name or RGB tuple.
            clear (bool): Whether to clear the buffer before setting the LEDs.
            brightness (float): Brightness level (0.0 to 1.0).
        """
        self.set_pixels(leds, color, clear)
        self.show(brightness)

        

    def set_circle(self, radius, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, filled=True):
        """
        Set a circle of LEDs to the specified color.

        Parameters:
            radius (float): Radius of the circle in LED units.
            offset_x (float): X-offset from the center of the array in LED units.
            offset_y (float): Y-offset from the center of the array in LED units.
            color (str or tuple): Color name or RGB tuple for the circle.
            half (str): 'top', 'bottom', 'left', 'right', or None (default).
            outside_color (str or tuple): Color name or RGB tuple for outside the circle.
            clear (bool): Whether to clear the buffer before drawing the circle.
            filled (bool): Whether to fill the circle or just draw the perimeter.
        """
        if clear: self.clear()
        cx, cy = 7.5 + offset_x, 7.5 + offset_y  # Use 7.5 to center in a 16x16 grid
        r, g, b = self._get_color(color)
        if outside_color:
            outside_r, outside_g, outside_b = self._get_color(outside_color)
        else:
            outside_r = outside_g = outside_b = 0

        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                fx, fy = x , y 
                distance_squared = (fx - cx) ** 2 + (fy - cy) ** 2
                inside_circle = distance_squared <= radius ** 2
                if filled:
                    if inside_circle:
                        if half == 'top':
                            if fy >= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'bottom':
                            if fy <= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'left':
                            if fx <= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'right':
                            if fx >= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half is None:
                            self._buf[x][y] = [r, g, b]
                        else:
                            raise ValueError(f"Invalid half specification: {half}")
                    elif outside_color:
                        self._buf[x][y] = [outside_r, outside_g, outside_b]
                else:
                    on_perimeter = radius ** 2 - radius < distance_squared < radius ** 2 + radius
                    if on_perimeter:
                        if half == 'top':
                            if fy >= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'bottom':
                            if fy <= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'left':
                            if fx <= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'right':
                            if fx >= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half is None:
                            self._buf[x][y] = [r, g, b]
                        else:
                            raise ValueError(f"Invalid half specification: {half}")
                    elif outside_color:
                        self._buf[x][y] = [outside_r, outside_g, outside_b]


    def show_circle(self, radius, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, brightness=None, filled=True):
        """
        Draw and show a circle of LEDs with the specified color.

        Parameters:
            radius (float): Radius of the circle in LED units.
            offset_x (float): X-offset from the center of the array in LED units.
            offset_y (float): Y-offset from the center of the array in LED units.
            color (str or tuple): Color name or RGB tuple for the circle.
            half (str): 'top', 'bottom', 'left', 'right', or None (default).
            outside_color (str or tuple): Color name or RGB tuple for outside the circle.
            clear (bool): Whether to clear the buffer before drawing the circle.
            brightness (float): Brightness level (0.0 to 1.0).
            filled (bool): Whether to fill the circle or just draw the perimeter.
        """
        self.set_circle(radius, offset_x, offset_y, color, half, outside_color, clear, filled)
        self.show(brightness)


    def set_square(self, side, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, filled=True):
        """
        Set a square of LEDs to the specified color.

        Parameters:
            side (float): Side length of the square in LED units.
            offset_x (float): X-offset from the center of the array in LED units.
            offset_y (float): Y-offset from the center of the array in LED units.
            color (str or tuple): Color name or RGB tuple for the square.
            half (str): 'top', 'bottom', 'left', 'right', or None (default).
            outside_color (str or tuple): Color name or RGB tuple for outside the square.
            clear (bool): Whether to clear the buffer before drawing the square.
            filled (bool): Whether to fill the square or just draw the perimeter.
        """
        if clear: self.clear()
        cx, cy = 7.5 + offset_x, 7.5 + offset_y  # Use 7.5 to center in a 16x16 grid
        r, g, b = self._get_color(color)
        half_side = side / 2
        start_x = cx - half_side
        end_x = cx + half_side
        start_y = cy - half_side
        end_y = cy + half_side
        
        if outside_color:
            outside_r, outside_g, outside_b = self._get_color(outside_color)
        else:
            outside_r = outside_g = outside_b = 0

        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                fx, fy = x, y 
                inside_square = start_x <= fx <= end_x and start_y <= fy <= end_y
                if filled:
                    if inside_square:
                        if half == 'top':
                            if fy >= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'bottom':
                            if fy <= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'left':
                            if fx <= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'right':
                            if fx >= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half is None:
                            self._buf[x][y] = [r, g, b]
                        else:
                            raise ValueError(f"Invalid half specification: {half}")
                    elif outside_color:
                        self._buf[x][y] = [outside_r, outside_g, outside_b]
                else:
                    on_perimeter = (
                        (start_x <= fx <= end_x and (fy == int(start_y) or fy == int(end_y))) or
                        (start_y <= fy <= end_y and (fx == int(start_x) or fx == int(end_x)))
                    )
                    on_perimeter = (
                        (start_x <= fx <= end_x and (abs(fy-start_y)<=.5 or abs(fy-end_y)<=0.5)) or
                        (start_y <= fy <= end_y and (abs(fx-start_x)<=0.5 or abs(fx-end_x)<=0.5))
                    )
                    on_perimeter = (
                        (start_x < fx < end_x and (abs(fy-start_y)<=0.5 or abs(fy-end_y)<=0.5)) or
                        (start_y < fy < end_y and (abs(fx-start_x)<=0.5 or abs(fx-end_x)<=0.5))
                    )
                    on_perimeter = (
                        (start_x <= fx <= end_x and (fy == int(start_y)+1 or fy == int(end_y))) or
                        (start_y <= fy <= end_y and (fx == int(start_x)+1 or fx == int(end_x)))
                    )
                    if on_perimeter:
                        if half == 'top':
                            if fy >= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'bottom':
                            if fy <= cy:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'left':
                            if fx <= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half == 'right':
                            if fx >= cx:
                                self._buf[x][y] = [r, g, b]
                        elif half is None:
                            self._buf[x][y] = [r, g, b]
                        else:
                            raise ValueError(f"Invalid half specification: {half}")
                    elif outside_color:
                        self._buf[x][y] = [outside_r, outside_g, outside_b]


    def show_square(self, side, offset_x=0, offset_y=0, color='white', half=None, outside_color=None, clear=True, brightness=None, filled=True):
        """
        Draw and show a square of LEDs with the specified color.

        Parameters:
            side (float): Side length of the square in LED units.
            offset_x (float): X-offset from the center of the array in LED units.
            offset_y (float): Y-offset from the center of the array in LED units.
            color (str or tuple): Color name or RGB tuple for the square.
            half (str): 'top', 'bottom', 'left', 'right', or None (default).
            outside_color (str or tuple): Color name or RGB tuple for outside the square.
            clear (bool): Whether to clear the buffer before drawing the square.
            brightness (float): Brightness level (0.0 to 1.0).
            filled (bool): Whether to fill the square or just draw the perimeter.
        """
        self.set_square(side, offset_x, offset_y, color, half, outside_color, clear, filled)
        self.show(brightness)

    def set_cross(self, thickness, span, color='white', offset_x=0, offset_y=0, clear=True):
        """
        Set a cross of LEDs to the specified color.

        Parameters:
            thickness (int): Thickness of the cross arms in LED units.
            span (float): Span of the cross arms in LED units.
            color (str or tuple): Color name or RGB tuple for the cross.
            offset_x (float): X-offset from the center of the array in LED units.
            offset_y (float): Y-offset from the center of the array in LED units.
            clear (bool): Whether to clear the buffer before drawing the cross.
        """
        if clear: self.clear()
        cx, cy = 7.5 + offset_x, 7.5 + offset_y  # Use 7.5 to center in a 16x16 grid
        r, g, b = self._get_color(color)
        
        half_thickness = thickness / 2
        start_x = cx - span / 2
        end_x = cx + span / 2
        start_y = cy - span / 2
        end_y = cy + span / 2
        
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                fx, fy = x , y 
                #if start_x <= fx <= end_x and cy - half_thickness <= fy <= cy + half_thickness:
                if start_x < fx < end_x and cy - half_thickness < fy < cy + half_thickness:
                    self._buf[x][y] = [r, g, b]
                # elif start_y <= fy <= end_y and cx - half_thickness <= fx <= cx + half_thickness:
                elif start_y <= fy < end_y and cx - half_thickness <= fx < cx + half_thickness:
                    self._buf[x][y] = [r, g, b]

    def show_cross(self, thickness, span, color='white', offset_x=0, offset_y=0, clear=True, brightness=None):
        """
        Draw and show a cross of LEDs with the specified color.

        Parameters:
            thickness (int): Thickness of the cross arms in LED units.
            span (float): Span of the cross arms in LED units.
            color (str or tuple): Color name or RGB tuple for the cross.
            offset_x (float): X-offset from the center of the array in LED units.
            offset_y (float): Y-offset from the center of the array in LED units.
            clear (bool): Whether to clear the buffer before drawing the cross.
            brightness (float): Brightness level (0.0 to 1.0).
        """
        self.set_cross(thickness, span, color, offset_x, offset_y, clear)
        self.show(brightness)

    def set_half(self, half, color='white', clear=True):
        """
        Set half of the LEDs to the specified color.

        Parameters:
            half (str): 'top', 'bottom', 'left', or 'right'.
            color (str or tuple): Color name or RGB tuple.
            clear (bool): Whether to clear the buffer before setting the half.
        """
        if clear: self.clear()
        r, g, b = self._get_color(color)
        if half == 'bottom':
            for y in range(self.HEIGHT // 2):
                for x in range(self.WIDTH):
                    self._buf[x][y] = [r, g, b]
        elif half == 'top':
            for y in range(self.HEIGHT // 2, self.HEIGHT):
                for x in range(self.WIDTH):
                    self._buf[x][y] = [r, g, b]
        elif half == 'left':
            for x in range(self.WIDTH // 2):
                for y in range(self.HEIGHT):
                    self._buf[x][y] = [r, g, b]
        elif half == 'right':
            for x in range(self.WIDTH // 2, self.WIDTH):
                for y in range(self.HEIGHT):
                    self._buf[x][y] = [r, g, b]
        else:
            raise ValueError(f"Invalid half specification: {half}")
                    
    def show_half(self, half, color='white', clear=True, brightness=None):
        """
        Draw and show half of the LEDs with the specified color.

        Parameters:
            half (str): 'top', 'bottom', 'left', or 'right'.
            color (str or tuple): Color name or RGB tuple.
            clear (bool): Whether to clear the buffer before setting the half.
            brightness (float): Brightness level (0.0 to 1.0).
        """
        self.set_half(half, color, clear)
        self.show(brightness)

    def clear(self):
        """
        Clear the buffer.
        """
        self._buf.fill(0)

    def set_all(self, r, g, b):
        """
        Set all pixels to the specified RGB color.

        Parameters:
            r (int): Red component (0-255).
            g (int): Green component (0-255).
            b (int): Blue component (0-255).
        """
        self._buf[:] = r, g, b

    def show(self, brightness=None):
        """
        Output the contents of the buffer to Unicorn HAT HD.

        Parameters:
            brightness (float): Brightness level (0.0 to 1.0).
        """
        if brightness is None: brightness = self._brightness
        self._spi.xfer2([self._SOF] + (np.rot90(self._buf, self._rotation).reshape(768) * brightness).astype(np.uint8).tolist())
        time.sleep(self._DELAY)

    def _get_color(self, color):
        """
        Get the RGB tuple for a predefined color or an RGB tuple.

        Parameters:
            color (str or tuple): Color name or RGB tuple.

        Returns:
            tuple: RGB color tuple.
        """
        if isinstance(color, str):
            return self.colors.get(color.lower(), (0, 0, 0))
        return color
    
    def check_leds_visual(self, brightness = 0.2):
        for c in ["red", "green", "blue", "white"]:
            self.on_all_color(c, brightness)
            reply = input(f"Press Enter if all LED are {c}...")
            
               
    def check_shapes(self, brightness = 0.2):
        self.show_circle(radius=4, color='green', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a filled green circle in the center of the grid)")

        self.show_circle(radius=4, offset_x=0.5, color='green', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a filled green circle shifted slightly to the right from the center)")

        self.show_circle(radius=4, color='green', half='top', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of a filled green circle in the center of the grid)")

        self.show_circle(radius=4, offset_x=0.5, color='green', half='top', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of a filled green circle shifted slightly to the right from the center)")

        self.show_circle(radius=4, color='blue', half='left', filled=False, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the left half of a blue circle's perimeter in the center of the grid)")

        self.show_circle(radius=4, offset_x=1, color='blue', half='left', outside_color='white', filled=False, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the left half of a blue circle's perimeter shifted to the right with the outside area colored white)")

        self.show_square(side=4, color='green', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a filled green square in the center of the grid)")

        self.show_square(side=5, color='green', outside_color='red', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a filled green square with a red border in the center of the grid)")

        self.show_square(side=4, color='green', half='top', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of a filled green square in the center of the grid)")

        self.show_square(side=5, color='green', half='top', outside_color='red', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of a filled green square with a red border in the center of the grid)")

        self.show_square(side=4, offset_x=3, color='blue', half='left', outside_color='white', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the left half of a filled blue square shifted to the right with a white border)")

        self.show_square(side=8, color='black', outside_color='white', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see an empty with a white border in the center of the grid)")

        self.show_square(side=8, color='blue', outside_color='white', filled=False, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a blue square's perimeter with a white inside in the center of the grid)")

        self.show_square(side=8, offset_x=2, color='blue', outside_color='white', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a filled blue square shifted to the right with a white border)")

        self.show_square(side=8, offset_x=2, color='blue', outside_color='white', filled=False, brightness=brightness)
        reply = input("Press Enter to continue... (You should see a blue square's perimeter shifted to the right with a white inside)")

        self.show_square(side=8, color='blue', outside_color='white', half='top', filled=True, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of a filled blue square with a white border in the center of the grid)")

        self.show_square(side=8, color='blue', outside_color='white', half='top', filled=False, brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of a blue square's perimeter with a white inside in the center of the grid)")

        self.show_cross(thickness=2, span=12, color='yellow', offset_x=0, offset_y=0)
        reply = input("Press Enter to continue... (You should see a yellow cross with a thickness of 2 and a span of 12 in the center of the grid)")

        self.show_pixel(x=10, y=10, color=(100, 0, 150))
        reply = input("Press Enter to continue... (You should see a single pixel at position (10, 10) with a color of RGB (100, 0, 150))")

        self.show_half('top', 'red', brightness=brightness)
        reply = input("Press Enter to continue... (You should see the top half of the grid filled with red color)")

        

# Example usage
if __name__ == '__main__':
    led_matrix = RPiLedMatrix()
    led_matrix.set_brightness(0.2)
    led_matrix.set_rotation(0)
    led_matrix.check_leds_visual()

    led_matrix.show_pixel(3,4)
    reply = input("Press Enter to continue... (You should see a white led in (3, 4))")
    
    leds_to_set = [(0, 0), (1, 1), (2, 2), (3, 3)]
    led_matrix.show_pixels(leds_to_set, color='blue', clear=True)
    reply = input("Press Enter to continue... (You should see a blue line of LEDs from (0, 0) to (3, 3))")
    
    
    led_matrix.off()
    led_matrix.close()
