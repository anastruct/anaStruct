import unittest

from anastruct.fem import system as se


class TestSystemElements(unittest.TestCase):
    def test_get_stiffness_matrix(self):
        # Create a system
        system = se.SystemElements()

        # Add a simple beam element (make sure it has a stiffness matrix)
        system.add_element(
            location=[[0, 0], [5, 0]]
        )  # Adjust as necessary based on your implementation

        # Here, we assume that the added element has a stiffness matrix defined
        # Check if stiffness matrix is correctly returned for element 1
        stiffness_matrix = system.get_stiffness_matrix(1)

        # Assert that a stiffness matrix was returned
        self.assertIsNotNone(stiffness_matrix)
        print("Stiffness matrix returned successfully.")

    def test_invalid_element(self):
        # Create a system without adding any elements
        system = se.SystemElements()

        # Try to get the stiffness matrix of an element that doesn't exist
        stiffness_matrix = system.get_stiffness_matrix(999)  # Nonexistent ID
        self.assertIsNone(stiffness_matrix)
        print("Handled invalid element ID correctly.")


if __name__ == "__main__":
    unittest.main()
