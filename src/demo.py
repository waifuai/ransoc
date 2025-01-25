# demo.py
from ransoc.search import hit
from ransoc.planet import Planet
from ransoc.star import Star

planets = []
search_embeddings = [[1, 1], [2, 2], [3, 3]]
for search_embedding in search_embeddings:
    current_planet = Planet(search_embedding)  # Correct instantiation
    planets.append(current_planet)

star = Star([4, 4])  # Correct instantiation
hit_planet, planets = hit(planets, star)

print("The hit planet was ", hit_planet)