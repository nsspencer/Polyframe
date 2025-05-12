import unittest
import polyframe

Transform = polyframe.define_convention(
    polyframe.Direction.FORWARD, polyframe.Direction.LEFT, polyframe.Direction.UP)


if __name__ == "__main__":
    unittest.main()
