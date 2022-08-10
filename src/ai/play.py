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

    def get_ascii_array(self, ascii: str):
        return ascii.replace("    ", "").split("\n")

    def get_string(self, content: str) -> str:
        return "".join(self.get_ascii_array(content)) + "\n"

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
