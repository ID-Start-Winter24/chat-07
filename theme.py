from gradio.themes.base import Base


class CustomTheme(Base):
    def __init__(self):
        super().__init__()

        # Customize the input field border color
        super().set(
            # Light mode gradient
            body_background_fill="linear-gradient(to top, #8e9eab, #eef2f3)",
            # Dark mode gradient
            body_background_fill_dark="linear-gradient(to top, #d7d2cc, #304352)",
            input_background_fill="#E5E4E2",  # Input background customization
            input_background_fill_dark="#E5E4E2",  # Input background for dark mode
        )
