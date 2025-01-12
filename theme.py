from gradio.themes.base import Base


class CustomTheme(Base):
    def __init__(self):
        super().__init__()

        # Customize the theme properties
        super().set(
            # Light mode
            body_background_fill="#ffffff",
            # Dark mode
            body_background_fill_dark="#ffffff",
            input_background_fill="#ffffff",  # Input background for light mode
            input_background_fill_dark="#ffffff"  # Input background for dark mode
        )
