"""
Basic test file to verify test infrastructure.
"""

def test_imports():
    """Test that we can import the main package."""
    import afet
    assert afet is not None

def test_version():
    """Test that we can get the package version."""
    import afet
    assert hasattr(afet, '__version__')
    assert isinstance(afet.__version__, str)
