class Col:
    def __init__(self):
        # text colours
        self.Black = "\033[30m"
        self.Red = "\033[31m"
        self.Green = "\033[32m"
        self.Yellow = "\033[33m"
        self.Blue = "\033[34m"
        self.Mag = "\033[35m"
        self.Cyan = "\033[36m"
        self.White = "\033[37m"
        # bright text colours
        self.black = "\033[90m"
        self.red = "\033[91m"
        self.green = "\033[92m"
        self.yellow = "\033[93m"
        self.blue = "\033[94m"
        self.mag = "\033[95m"
        self.cyan = "\033[96m"
        self.white = "\033[97m"
        self.text_reset = "\033[0m"
        # background colours
        self.bg_black = "\033[40m"
        self.bg_red = "\033[41m"
        self.bg_green = "\033[42m"
        self.bg_yellow = "\033[43m"
        self.bg_blue = "\033[44m"
        self.bg_mag = "\033[45m"
        self.bg_cyan = "\033[46m"
        self.bg_white = "\033[47m"
        self.bg_reset = "\033[49m"

        self.bold = "\033[1m"  # Bold on
        self.faint = "\033[2m"  # Faint off
        self.italics = "\033[3m"  # Italic on
        self.underline = "\033[4m"  # Underline on
        self.blink_slow = "\033[5m"  # Slow blink on
        self.blink_fast = "\033[6m"  # Rapid blink on
        self.flash = self.blink_fast  # Flash on
        self.inv = "\033[7m"  # invert colours
        self.conceal = "\033[8m"  # Conceal on
        self.crossed_out = "\033[9m"  # Crossed-out on
        self.bold_off = "\033[22m"  # Bold off
        self.italics_off = "\033[23m"  # Italic off
        self.underline_off = "\033[24m"  # Underline off
        self.blink_off = "\033[25m"  # Blink off
        self.flash_off = self.blink_off  # Flash off
        self.reverse_off = "\033[27m"  # Reverse video off
        self.conceal_off = "\033[28m"  # Conceal off
        self.crossed_out_off = "\033[29m"  # Crossed-out off

        self.reverse = "\033[7m"
        self.hidden = "\033[8m"

        self.strikethrough = "\033[9m"
        # reset all text cols
        self.end = (
            self.text_reset
            + self.bg_reset
            + self.bold_off
            + self.italics_off
            + self.underline_off
            + self.blink_off
            + self.flash_off
            + self.reverse_off
            + self.conceal_off
            + self.crossed_out_off
        )
        self.reset = self.end


col = Col()
