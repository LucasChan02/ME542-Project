import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

path_to_step = 'scan2_volume_v5.step'
gmsh.model.occ.importShapes(path_to_step)

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

bbox = gmsh.model.getBoundingBox(-1, -1)
print(f"Model Bounding Box: {bbox}")

volumes = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)

if not volumes:
    print("Error: No volume found in STEP file.")
    sys.exit()

vol_tags = [v[1] for v in volumes]
p_vol = gmsh.model.addPhysicalGroup(3, vol_tags, name="Internal_Volume")

surf_tags = [s[1] for s in surfaces]
p_surf = gmsh.model.addPhysicalGroup(2, surf_tags, name="Outer_Skin")

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)

gmsh.model.mesh.generate(3)

gmsh.write("output_mesh.msh")

gmsh.finalize()