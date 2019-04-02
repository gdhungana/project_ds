from graph import Graph

graph1 = { "a" : ["c",'z'],
          "b" : ["c", "e"],
          "c" : ["a", "b", "d", "e"],
          "d" : ["c"],
          "e" : ["c", "b"],
          "f" : []
         }

grap=Graph(graph1)
print("Vertices: {}".format(grap.vertices()))
print("Edges: ", grap.edges())
grap.add_vertex('x')
print("Vertices 2:", grap.vertices())
grap.add_edge(('x','e'))
print("Graph:", grap)
print("Edges 2: ", grap.edges())

print("Path a--> e:", grap.find_path('a','e'))

print("Paths a--> e:", grap.find_all_paths('a','e'))
print("Paths c--> z:", grap.find_all_paths('c','z'))
