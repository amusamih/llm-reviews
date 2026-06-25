def test_server_module_imports_without_flask_side_effects():
    from app.server import create_app

    assert callable(create_app)
