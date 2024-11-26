from gradio.themes.base import Base


class CustomTheme(Base):
    def __init__(self):
        super().__init__()

        super().set(
            body_background_fill="#CDCDCB",  # Fallback color
            body_background_fill_dark="black",
            input_background_fill="#E5E4E2",
            input_background_fill_dark="#E5E4E2"
        )
