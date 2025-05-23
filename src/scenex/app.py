def run() -> None:
    """Enter the native GUI event loop."""
    from scenex.adaptors import determine_backend

    if determine_backend() == "vispy":
        from vispy.app import run

        run()
    else:
        from rendercanvas.auto import loop

        loop.run()
