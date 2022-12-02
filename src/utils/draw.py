from kniffel.classes.kniffel import Kniffel
from kniffel.classes.options import KniffelOptions


class KniffelDraw:
    """
    Draw Helper class to create output
    """

    dice_1 = """-----
                |   |
                | o |
                |   |
                -----"""

    dice_2 = """-----
                |o  |
                |   |
                |  o|
                -----"""

    dice_3 = """-----
                |o  |
                | o |
                |  o|
                -----"""

    dice_4 = """-----
                |o o|
                |   |
                |o o|
                -----"""

    dice_5 = """-----
                |o o|
                | o |
                |o o|
                -----"""

    dice_6 = """-----
                |o o|
                |o o|
                |o o|
                -----"""

    _game_log = []

    def __init__(self):
        """Create new draw helper"""
        pass

    def get_ascii_array(self, ascii: str) -> list:
        """Return ascii string as list of strings

        :param ascii: ascii object
        :return: list of ascii object
        """
        return ascii.replace("    ", "").split("\n")

    def get_string(self, content: str) -> str:
        """Return string with line breaks added

        :param content: content
        :return: string with line breaks added
        """
        return "".join(self.get_ascii_array(content)) + "\n"

    def draw_sheet(self, kniffel: Kniffel) -> str:
        """Return drawn kniffel sheet based on the current state from kniffel

        :param kniffel: Kniffel game object
        :return: kniffel sheet
        """

        state = kniffel.get_state()[0]

        sheet = f"""     ____________________________________      ____________________________________
    | CATEGORY       | STATE  |  POINTS  |    | CATEGORY       | STATE  |  POINTS  |
    |::::::::::::::::|::::::::|::::::::::|    |::::::::::::::::|::::::::|::::::::::|
    | ONES           |   {self.get_float(state[34], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.ONES, KniffelOptions.ONES_SLASH))}    |    | THREE TIMES    |   {self.get_float(state[40], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.THREE_TIMES, KniffelOptions.THREE_TIMES_SLASH))}    |
    |================|========|==========|    |================|========|==========|
    | TWOS           |   {self.get_float(state[35], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.TWOS, KniffelOptions.TWOS_SLASH))}    |    | FOURS TIMES    |   {self.get_float(state[41], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.FOUR_TIMES, KniffelOptions.FOUR_TIMES_SLASH))}    |
    |================|========|==========|    |================|========|==========|
    | THREES         |   {self.get_float(state[36], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.THREES, KniffelOptions.THREES_SLASH))}    |    | FULL HOUSE     |   {self.get_float(state[42], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.FULL_HOUSE, KniffelOptions.FULL_HOUSE_SLASH))}    |
    |================|========|==========|    |================|========|==========|
    | FOURS          |   {self.get_float(state[37], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.FOURS, KniffelOptions.FOURS_SLASH))}    |    | SMALL STREET   |   {self.get_float(state[43], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.SMALL_STREET, KniffelOptions.SMALL_STREET_SLASH))}    |
    |================|========|==========|    |================|========|==========|
    | FIVES          |   {self.get_float(state[38], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.FIVES, KniffelOptions.FIVES_SLASH))}    |    | LARGE STREET   |   {self.get_float(state[44], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.LARGE_STREET, KniffelOptions.LARGE_STREET_SLASH))}    |
    |================|========|==========|    |================|========|==========|
    | SIXES          |   {self.get_float(state[39], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.SIXES, KniffelOptions.SIXES_SLASH))}    |    | KNIFFEL        |   {self.get_float(state[45], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.KNIFFEL, KniffelOptions.KNIFFEL_SLASH))}    |
    |################|########|##########|    |================|========|==========|
    |  BONUS         |   {self.get_float(state[33], state=True)}   |    {self.get_float(35 if kniffel.is_bonus() else 0)}    |    | CHANCE         |   {self.get_float(state[46], state=True)}   |    {self.get_float(kniffel.get_option_kniffel_points(KniffelOptions.CHANCE, KniffelOptions.CHANCE_SLASH))}    |
    |================|========|==========|    |################|########|##########|
    |  TOP POINTS    |        |   {self.get_long_float(kniffel.get_points_top())}    |    |  BOTTOM POINTS |        |   {self.get_long_float(kniffel.get_points_bottom())}    |
    |################|########|##########|    |================|========|==========|
                                              |  TOTAL POINTS  |        |   {self.get_long_float(kniffel.get_points())}    |
                                              |################|########|##########|"""

        return sheet

    def get_long_float(self, f: float) -> str:
        """Return float as a fixed size as string. Length: 3 chars

        :param f: floar
        :return: float transfered to string
        """

        if f < 10:
            return "  " + str(f)
        elif f < 100:
            return " " + str(f)
        elif f < 1000:
            return str(f)

    def get_float(self, f: float, dec=0, state=False) -> str:
        s = str(f)

        if state:
            if f >= 0:
                return " 1"
            else:
                return "-1"

        if f == -1:
            return "-1"
        elif f == 0 and dec == 0:
            return " 0"
        elif f == 0 and dec == 3:
            return "   0 "
        elif f == 1:
            return " 1"
        elif f == -10:
            return " 0"
        elif f < 10:
            return " " + str(f)
        else:
            return str(round(f, dec))

    def get_dice(self, dice_array: list) -> int:
        dice = 0
        counter = 0

        for val in dice_array:
            counter += 1
            if val == 1:
                dice = counter
                break
        return dice

    def draw_dices(self, dice_array: list) -> str:
        """Return dices as ascii art

        :param dice_array: list of dices
        :return: ascii art as string
        """
        lines = {}
        lines[0] = []
        lines[1] = []
        lines[2] = []
        lines[3] = []
        lines[4] = []

        dices = [
            self.get_dice(dice_array[0:6]),
            self.get_dice(dice_array[6:12]),
            self.get_dice(dice_array[12:18]),
            self.get_dice(dice_array[18:24]),
            self.get_dice(dice_array[24:30]),
        ]

        for dice_scaled in dices:
            dice = round(dice_scaled)

            ascii = ""
            if dice == 1:
                ascii = self.dice_1
            if dice == 2:
                ascii = self.dice_2
            if dice == 3:
                ascii = self.dice_3
            if dice == 4:
                ascii = self.dice_4
            if dice == 5:
                ascii = self.dice_5
            if dice == 6:
                ascii = self.dice_6

            counter = 0
            for l in self.get_ascii_array(ascii):
                lines[counter].append(l + "     ")

                counter += 1

        result = ""
        for c in range(5):
            lines[c].insert(0, "    ")
            result += "".join(lines[c]) + "\n"

        return result

    def draw_kniffel_title(self) -> str:
        """Return kniffel as ascii art

        :param return: kniffel ascii art
        """
        return """
 __  ___ .__   __.  __   _______  _______  _______  __                    .______       __      
|  |/  / |  \ |  | |  | |   ____||   ____||   ____||  |                   |   _  \     |  |     
|  '  /  |   \|  | |  | |  |__   |  |__   |  |__   |  |         ______    |  |_)  |    |  |     
|    <   |  . `  | |  | |   __|  |   __|  |   __|  |  |        |______|   |      /     |  |     
|  .  \  |  |\   | |  | |  |     |  |     |  |____ |  `----.              |  |\  \----.|  `----.
|__|\__\ |__| \__| |__| |__|     |__|     |_______||_______|              | _| `._____||_______|
                                                                                                
"""


if __name__ == "__main__":
    draw = KniffelDraw()

    result = draw.draw_dices(
        [0.66666667, 0.5, 0.16666667, 0.66666667, 0.66666667, 0.33333333]
    )

    print(result)
