from gradio.themes.base import Base


class CustomTheme(Base):
    def __init__(self):
        super().__init__()

        # Customize the theme properties
        super().set(
            # Light mode
            body_background_fill="#ffffff",
            # Dark mode
            input_background_fill="#ffffff",  # Input background for light mode
        )
