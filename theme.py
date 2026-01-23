from gradio.themes.base import Base


class CustomTheme(Base):
    def __init__(self):
        super().__init__()

        # Customize the theme properties
        super().set(
            body_background_fill="#ffffff",
            input_background_fill="#ffffff",
            input_border_color="transparent",
            input_border_width="0px",
            input_shadow="none",
            input_shadow_focus="none",
        )
