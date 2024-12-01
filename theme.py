from gradio.themes.base import Base


class CustomTheme(Base):
    def __init__(self):
        super().__init__()

        # Customize the theme properties
        super().set(
            # Light mode
            body_background_fill="linear-gradient(to top, #8e9eab, #eef2f3)",
            # Dark mode
            body_background_fill_dark="linear-gradient(to top, #434343, #000000)",
            input_background_fill="#e8e9ea",  # Input background for light mode
            input_background_fill_dark="#2C2C2C"  # Input background for dark mode
        )
