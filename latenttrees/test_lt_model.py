import unittest
from unittest.mock import MagicMock
import latenttrees.lt_model as lt


class TestStructureUpdate(unittest.TestCase):
    def test_something(self):
        g = lt.Graph()
        g.register_properties(set(), {'x'}, set())
        data = MagicMock(spec=lt.Data)
        parameter_learning = MagicMock(spec=lt.BeliefPropagation)
        structure_update = lt.StructureUpdateSampling(g, data, parameter_learning)

        # build
        id_nodes = g.add_nodes([2, 2, 2])
        g.add_edge(2, 0)
        g.add_edge(2, 1)

        structure_update._StructureUpdate__remove_old_parent(0)

        # self.assertEqual(True, False)
        self.assertFalse(g.has_node(2))
        self.assertTrue(g.has_node(0))
        self.assertTrue(g.has_node(1))

if __name__ == '__main__':
    unittest.main()
