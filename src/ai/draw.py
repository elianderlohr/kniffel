class KniffelDraw:

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
        self._game_log.append(self.get_string(self.draw_kniffel_title()))

    def append_log(self, content: str):
        self._game_log.append(self.get_string(content))

    def get_ascii_array(self, ascii: str):
        return ascii.replace("    ", "").split("\n")

    def get_string(self, content: str) -> str:
        return "".join(self.get_ascii_array(content)) + "\n"

    def draw_sheet(self, state):
        sheet = f"""     ____________________________________      ____________________________________
    | CATEGORY       | STATE  |  POINTS  |    | CATEGORY       | STATE  |  POINTS  |
    |::::::::::::::::|::::::::|::::::::::|    |::::::::::::::::|::::::::|::::::::::|
    | ONES           |   {self.get_float(state[1])}   |    20    |    | THREE TIMES    |   {self.get_float(state[9])}   |    20    |
    |================|========|==========|    |================|========|==========|
    | TWOS           |   {self.get_float(state[2])}   |    20    |    | FOURS TIMES    |   {self.get_float(state[10])}   |    20    |
    |================|========|==========|    |================|========|==========|
    | THREES         |   {self.get_float(state[3])}   |    20    |    | FULL HOUSE     |   {self.get_float(state[11])}   |    20    |
    |================|========|==========|    |================|========|==========|
    | FOURS          |   {self.get_float(state[4])}   |    20    |    | SMALL STREET   |   {self.get_float(state[12])}   |    20    |
    |================|========|==========|    |================|========|==========|
    | FIVES          |   {self.get_float(state[5])}   |    20    |    | LARGE STREET   |   {self.get_float(state[13])}   |    20    |
    |================|========|==========|    |================|========|==========|
    | SIXES          |   {self.get_float(state[6])}   |    20    |    | KNIFFEL        |   {self.get_float(state[14])}   |    20    |
    |################|########|##########|    |================|========|==========|
    |  BONUS         |   {self.get_float(state[7])}   |    20    |    | CHANCE         |   {self.get_float(state[15])}   |    20    |
    |================|========|==========|    |################|########|##########|
    |  TOP POINTS    | {self.get_float(state[8], 3)}  |    20    |    |  BOTTOM POINTS | {self.get_float(state[16], 3)}  |    20    |
    |################|########|##########|    |================|========|==========|
                                              |  TOTAL POINTS  | {self.get_float(state[17], 3)}  |    20    |
                                              |################|########|##########|"""

        return sheet

    def get_float(self, f: float, dec = 0) -> str:
        s = str(f)

        
        if f == -1:
            return "-1"
        elif f == 0 and dec == 0:
            return " 0"
        elif f == 0 and dec == 3:
            return "   0 "
        elif f == 1:
            return " 1"
        else:
            return str(round(f, dec))

    def draw_dices(self, dice_array: []):
        lines = {}
        lines[0] = []
        lines[1] = []
        lines[2] = []
        lines[3] = []
        lines[4] = []
        for dice_scaled in dice_array:
            dice = round(dice_scaled * 6)

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
            lines[c].insert(0, "                        ")
            result += "".join(lines[c]) + "\n"

        return result

    def draw_kniffel_title(self):
        return """
 _  ___   _ _____ ______ ______ ______ _      
| |/ / \ | |_   _|  ____|  ____|  ____| |     
| ' /|  \| | | | | |__  | |__  | |__  | |     
|  < | . ` | | | |  __| |  __| |  __| | |     
| . \| |\  |_| |_| |    | |    | |____| |____ 
|_|\_\_| \_|_____|_|    |_|    |______|______|"""

    def draw_title(self, title: str):
        s = f"""
        #######################################
        #######################################
        ###### {title} ########################
        #######################################
        #######################################
        """


if __name__ == "__main__":
    draw = KniffelDraw()

    result = draw.draw_dices(
        [0.66666667, 0.5, 0.16666667, 0.66666667, 0.66666667, 0.33333333]
    )

    print(result)
