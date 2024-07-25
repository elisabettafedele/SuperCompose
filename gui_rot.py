# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import json
from threading import Event
from queue import Queue


isMacOS = (platform.system() == "Darwin")
OUTPUT_JSON = "objects.json"

INSTANCE_COLORS = {
0:  (165.0, 80.0, 115.0),
1: (254., 97., 0.), #orange
2: (120., 94., 240.), #purple 
3: (100., 143., 255.), #blue
4: (220., 38., 127.), #pink
5: (255., 176., 0.), #yellow
6: (100., 143., 255.), 
7: (160.0, 50.0, 50.0), 
8:  (129.0, 0.0, 50.0), 
9:  (255., 176., 0.), 
10: (192.0, 100.0, 119.0), 
11: (149.0, 192.0, 228.0), 
12: (14.0, 0.0, 120.0), 
13: (90., 64., 210.), 
14: (152.0, 200.0, 156.0),
15: (129.0, 103.0, 106.0), 
16: (100.0, 160.0, 100.0),  #
17: (70.0, 70.0, 140.0), 
18: (160.0, 20.0, 60.0), 
19: (20., 130., 20.), 
20: (140.0, 30.0, 60.0),
21:  (20.0, 20.0, 120.0), 
22:  (243.0, 115.0, 68.0), 
23:  (120.0, 162.0, 227.0), 
24:  (100.0, 78.0, 142.0), 
25:  (152.0, 95.0, 163.0), 
26:  (160.0, 20.0, 60.0), 
27:  (100.0, 143.0, 255.0), 
28: (255., 204., 153.),
29: (50., 100., 0.),
}

def load_superquadrics(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    superquadrics = []
    for component in data['components']:
        sq = Superquadric()
        sq.set_values(
            scale=component['scale'],
            position=component['position'],
            epsilon1=component['epsilon1'],
            epsilon2=component['epsilon2'],
            rotation=component['rotation']
        )
        superquadrics.append(sq)
    return superquadrics


def generate_superquadric_mesh(a1, a2, a3, epsilon1, epsilon2, num_points=50):
    phi, theta = np.meshgrid(
        np.linspace(-np.pi / 2, np.pi / 2, num_points),
        np.linspace(-np.pi, np.pi, num_points)
    )
    x = a1 * np.sign(np.cos(phi)) * np.abs(np.cos(phi))**epsilon1 * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**epsilon2
    y = a2 * np.sign(np.cos(phi)) * np.abs(np.cos(phi))**epsilon1 * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**epsilon2
    z = a3 * np.sign(np.sin(phi)) * np.abs(np.sin(phi))**epsilon1

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    faces = []
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            idx = i * num_points + j
            faces.append([idx, idx + 1, idx + num_points])
            faces.append([idx + 1, idx + num_points + 1, idx + num_points])
    
    return vertices, np.array(faces)


class Superquadric:
    def __init__(self):
        self.a1 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.a2 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.a3 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.e1 = gui.Slider(gui.Slider.DOUBLE)
        self.e1.set_limits(0.1, 1.9)
        self.e2 = gui.Slider(gui.Slider.DOUBLE)
        self.e2.set_limits(0.1, 1.9)
        self.x = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.y = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.z = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r11 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r12 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r13 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r21 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r22 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r23 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r31 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r32 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r33 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    
    def set_values(self, scale, position, epsilon1, epsilon2, rotation):
        self.a1.set_value(scale[0])
        self.a2.set_value(scale[1])
        self.a3.set_value(scale[2])
        self.e1 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.e2 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.e1.set_value(epsilon1)
        self.e2.set_value(epsilon2)
        self.x.set_value(position[0])
        self.y.set_value(position[1])
        self.z.set_value(position[2])
        self.r11.set_value(rotation[0][0])
        self.r12.set_value(rotation[0][1])
        self.r13.set_value(rotation[0][2])
        self.r21.set_value(rotation[1][0])
        self.r22.set_value(rotation[1][1])
        self.r23.set_value(rotation[1][2])
        self.r31.set_value(rotation[2][0])
        self.r32.set_value(rotation[2][1])
        self.r33.set_value(rotation[2][2])

        
    
class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21
    MENU_CAPTURE = 31
    MENU_STOP_CAPTURE = 32

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height, load):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        bounds = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        self._scene.setup_camera(60, bounds, bounds.get_center())
        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        view_ctrls.set_is_open(False)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 20)
    
        self._point_size.set_on_value_changed(self._on_point_size)
        #self._on_shader(None, 1) # Set unlit
        self._on_point_size(5)
        


        #h = gui.Horiz(0.25 * em)  # last rows - query


        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)
        material_settings.set_is_open(False)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        
        
        
        # Viz mode
        # viz_mode_selection_ctrls = gui.CollapsableVert("Visualization Mode", 0.25 * em,
        #                                  gui.Margins(em, 0, 0, 0))
        # self._viz_mode_list = gui.Combobox()
        
        # UNTIL HERE Standard settings panel
        
        
        # for name in ["RGB", "Similarity", "Similarity (Raw)", "Retrieved Instances"]:
        #     self._viz_mode_list.add_item(name)
        # self._viz_mode_list.set_on_selection_changed(self._on_viz_mode)

        # h = gui.Horiz(0.25 * em)  # last rows - scene selection
        # h.add_stretch()
        # h.add_child(gui.Label("Mode"))
        # h.add_child(self._viz_mode_list)
        # h.add_stretch()
        # viz_mode_selection_ctrls.add_child(h)

        #self._settings_panel.add_fixed(separation_height)
        # self._settings_panel.add_child(viz_mode_selection_ctrls)
        
                
        # Scene selection
        #scene_selection_ctrls = gui.Vert("Scene selection", 0, gui.Margins(em, 0, 0, 0))
        # scene_selection_ctrls = gui.CollapsableVert("Scene selection", 0.25 * em,
        #                                  gui.Margins(em, 0, 0, 0))
        
        # self.scene0011_00 = gui.Button("scene0011_00")
        # self.scene0011_00.horizontal_padding_em, self.scene0011_00.vertical_padding_em = 0.5, 0
        # self.scene0011_00.set_on_clicked(self._set_scene0011_00)

        # self.scene0458_01 = gui.Button("scene0458_01")
        # self.scene0458_01.horizontal_padding_em, self.scene0458_01.vertical_padding_em = 0.5, 0
        # self.scene0458_01.set_on_clicked(self._set_scene0458_01)

        # h = gui.Horiz(0.25 * em)  # last rows - scene selection
        # h.add_stretch()
        # h.add_child(self.scene0011_00)
        # h.add_child(self.scene0458_01)
        # h.add_stretch()
        # scene_selection_ctrls.add_child(h)

        # self.scene0549_00 = gui.Button("scene0549_00")
        # self.scene0549_00.horizontal_padding_em, self.scene0549_00.vertical_padding_em = 0.5, 0
        # self.scene0549_00.set_on_clicked(self._set_scene0549_00)

        # self.scene0565_00 = gui.Button("scene0565_00")
        # self.scene0565_00.horizontal_padding_em, self.scene0565_00.vertical_padding_em = 0.5, 0
        # self.scene0565_00.set_on_clicked(self._set_scene0565_00)

        # h = gui.Horiz(0.25 * em)  # last rows - scene selection
        # h.add_stretch()
        # h.add_child(self.scene0549_00)
        # h.add_child(self.scene0565_00)
        # h.add_stretch()
        # scene_selection_ctrls.add_child(h)


        self._settings_panel.add_fixed(separation_height)
        # self._settings_panel.add_child(scene_selection_ctrls)
        self.sqs = []
        self.sqs.append(Superquadric())
        self.stored_superquadrics = gui.CollapsableVert("Stored Superquadrics", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        self.stored_superquadrics.set_is_open(True)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.stored_superquadrics)
        
        self.textbox = gui.TextEdit()
        self.textbox.placeholder_text = "Enter your description here"
        h = gui.Horiz(0.25 * em)  # last rows - scene selection
        h.add_stretch()
        h.add_child(gui.Label("Description:"))
        h.add_child(self.textbox)
        h.add_stretch()
        self.stored_superquadrics.add_child(h)
        if(load):
            sqs = load_superquadrics("example.json")
        
            for idx, sq in enumerate(sqs):
                self.sqs.append(sq)
                em = self.window.theme.font_size
                sq = gui.CollapsableVert(f"SQ{idx+1}", 0.25 * em,
                                        gui.Margins(em, 0, 0, 0))
                grid = gui.VGrid(2, 0.25 * em)
                grid.add_child(gui.Label("Shape 1"))
                grid.add_child(self.sqs[-1].e1)
                grid.add_child(gui.Label("Shape 2"))
                grid.add_child(self.sqs[-1].e2)
                grid.add_child(gui.Label("Semi-axis 1"))
                grid.add_child(self.sqs[-1].a1)
                grid.add_child(gui.Label("Semi-axis 2"))
                grid.add_child(self.sqs[-1].a2)
                grid.add_child(gui.Label("Semi-axis 3"))
                grid.add_child(self.sqs[-1].a3)
                grid.add_child(gui.Label("x"))
                grid.add_child(self.sqs[-1].x)
                grid.add_child(gui.Label("y"))
                grid.add_child(self.sqs[-1].y)
                grid.add_child(gui.Label("z"))
                grid.add_child(self.sqs[-1].z)
                grid.add_child(gui.Label("r11"))
                grid.add_child(self.sqs[-1].r11)
                grid.add_child(gui.Label("r12"))
                grid.add_child(self.sqs[-1].r12)
                grid.add_child(gui.Label("r13"))
                grid.add_child(self.sqs[-1].r13)
                grid.add_child(gui.Label("r21"))
                grid.add_child(self.sqs[-1].r21)
                grid.add_child(gui.Label("r22"))
                grid.add_child(self.sqs[-1].r22)
                grid.add_child(gui.Label("r23"))
                grid.add_child(self.sqs[-1].r23)
                grid.add_child(gui.Label("r31"))
                grid.add_child(self.sqs[-1].r31)
                grid.add_child(gui.Label("r32"))
                grid.add_child(self.sqs[-1].r32)
                grid.add_child(gui.Label("r33"))
                grid.add_child(self.sqs[-1].r33)
                sq.add_child(grid)
                sq.set_is_open(True)
                self.stored_superquadrics.add_child(sq)
        
        else: 
            sq1 = gui.CollapsableVert("SQ1", 0.25 * em,
                                            gui.Margins(em, 0, 0, 0))
            
            grid = gui.VGrid(2, 0.25 * em)
            grid.add_child(gui.Label("Shape 1"))
            grid.add_child(self.sqs[0].e1)
            grid.add_child(gui.Label("Shape 2"))
            grid.add_child(self.sqs[0].e2)
            grid.add_child(gui.Label("Semi-axis 1"))
            grid.add_child(self.sqs[0].a1)
            grid.add_child(gui.Label("Semi-axis 2"))
            grid.add_child(self.sqs[0].a2)
            grid.add_child(gui.Label("Semi-axis 3"))
            grid.add_child(self.sqs[0].a3)
            grid.add_child(gui.Label("x"))
            grid.add_child(self.sqs[0].x)
            grid.add_child(gui.Label("y"))
            grid.add_child(self.sqs[0].y)
            grid.add_child(gui.Label("z"))
            grid.add_child(self.sqs[0].z)
            grid.add_child(gui.Label("roll"))
            grid.add_child(self.sqs[0].roll)
            grid.add_child(gui.Label("pitch"))
            grid.add_child(self.sqs[0].pitch)
            grid.add_child(gui.Label("yaw"))
            grid.add_child(self.sqs[0].yaw)
            sq1.add_child(grid)
            sq1.set_is_open(True)

            self._settings_panel.add_fixed(separation_height)
            self.stored_superquadrics.add_child(sq1)
        
        
        
        
        # Query
        # sq_parameters = gui.CollapsableVert("Try superquadric parameters:", 0.25 * em,
        #                                  gui.Margins(em, 0, 0, 0))
        
        
        # self.e1 = gui.Slider(gui.Slider.DOUBLE)
        # self.e1.set_limits(0.1, 1.9)
        # #self.e1.set_on_value_changed(self.visualize_superquadric_bis)
        # self.e2 = gui.Slider(gui.Slider.DOUBLE)
        # self.e2.set_limits(0.1, 1.9)
        # self.a1 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        # # self.a1.set_limits(0, 1)
        # self.a2 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        # # self.a2.set_limits(0, 1)
        # self.a3 = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        # # self.a3.set_limits(0, 1)
        # self.x = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        # self.y = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        # self.z = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        
        # grid = gui.VGrid(2, 0.25 * em)
        # grid.add_child(gui.Label("Shape 1"))
        # grid.add_child(self.e1)
        # grid.add_child(gui.Label("Shape 2"))
        # grid.add_child(self.e2)
        # grid.add_child(gui.Label("Semi-axis 1"))
        # grid.add_child(self.a1)
        # grid.add_child(gui.Label("Semi-axis 2"))
        # grid.add_child(self.a3)
        # grid.add_child(gui.Label("Semi-axis 3"))
        # grid.add_child(self.a2)
        # grid.add_child(gui.Label("x"))
        # grid.add_child(self.x)
        # grid.add_child(gui.Label("y"))
        # grid.add_child(self.y)
        # grid.add_child(gui.Label("z"))
        # grid.add_child(self.z)
        # sq_parameters.add_child(grid)
        # sq_parameters.set_is_open(True)

        # self._settings_panel.add_fixed(separation_height)
        # self._settings_panel.add_child(sq_parameters)
        
        self.visualize_button = gui.Button("Visualize")
        self.visualize_button.set_on_clicked(self.visualize_superquadrics)
        self.add_button = gui.Button("Add")
        self.add_button.set_on_clicked(self.add_superquadric)
        self.save_button = gui.Button("Save")
        self.save_button.set_on_clicked(self.save_object)
        
        self._settings_panel.add_child(self.visualize_button)
        self._settings_panel.add_child(self.add_button)
        self._settings_panel.add_child(self.save_button)
        


        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            capture_menu = gui.Menu()
            capture_menu.add_item("Start Capturing", AppWindow.MENU_CAPTURE)
            capture_menu.add_item("Stop Capturing", AppWindow.MENU_STOP_CAPTURE)
            


            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Capture", capture_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Capture", capture_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self.capture_t = None
        self.drawing_t = None
        self.close_event = Event()
        self.capture_buffer = Queue()
        self.pcd = None
        #set to true when drawing the first pcd after start capture is pressed. Used to center the camera on the pcd
        self._first_pcd = True
        self.scene_name = None
        self.device = o3d.core.Device("CUDA:0")
        self.dtype = o3d.core.float32
        #change to True if a cuda device is not available
        self.legacy_pcd = True # False
        #change to True to print computational times
        self.print_perf = False
        self.viz_mode = "RGB" # "Similarity", "Similarity (Raw)", "Retrieved Instances"
        self.query_text = ""
        self.no_query_yet = True
        self.instances_marker = []
        self.rescale_scores = True
        
        #slow down the pointcloud creation from the sensor
        self.filter_pcd_outliers = False

        self._apply_settings()
    
    def visualize_superquadrics(self):
        self._scene.scene.clear_geometry()
        for instance in self.instances_marker:
            self._scene.scene.add_geometry(f"__model_{id(instance)}__", instance, self.settings.material)

        for i in range(len(self.sqs)):
            sq = self.sqs[i]
            e1 = sq.e1.double_value
            e2 = sq.e2.double_value
            a1 = sq.a1.double_value
            a2 = sq.a2.double_value
            a3 = sq.a3.double_value
            if a1 * a2 * a3 != 0:
                x = sq.x.double_value
                y = sq.y.double_value
                z = sq.z.double_value
                r11 = sq.r11.double_value
                r12 = sq.r12.double_value
                r13 = sq.r13.double_value
                r21 = sq.r21.double_value
                r22 = sq.r22.double_value
                r23 = sq.r23.double_value
                r31 = sq.r31.double_value
                r32 = sq.r32.double_value
                r33 = sq.r33.double_value

                vertices, faces = generate_superquadric_mesh(a1, a2, a3, e1, e2)

                # Apply rotation using Euler angles
                R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
                vertices = vertices @ R.T
                vertices += np.array([x, y, z])
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([1, 0.706, 0])

                self._scene.scene.add_geometry(f"__superquadric__{i}", mesh, self.settings.material)

        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())


    # def visualize_superquadrics(self):
    #         self._scene.scene.clear_geometry()
    #         for instance in self.instances_marker:
    #             self._scene.scene.add_geometry(f"__model_{id(instance)}__", instance, self.settings.material)
            
    #         for i in range(len(self.sqs)):
    #             sq = self.sqs[i]
    #             e1 = sq.e1.double_value
    #             e2 = sq.e2.double_value
    #             a1 = sq.a1.double_value
    #             a2 = sq.a2.double_value
    #             a3 = sq.a3.double_value
    #             if( a1*a2*a3 != 0):
    #                 x = sq.x.double_value
    #                 y = sq.y.double_value
    #                 z = sq.z.double_value
                
    #                 vertices, faces = generate_superquadric_mesh(a1, a2, a3, e1, e2)
    #                 vertices += np.array([x, y, z])
    #                 mesh = o3d.geometry.TriangleMesh()
    #                 mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #                 mesh.triangles = o3d.utility.Vector3iVector(faces)
    #                 mesh.compute_vertex_normals()
    #                 mesh.paint_uniform_color([1, 0.706, 0])
            
    #                 self._scene.scene.add_geometry(f"__superquadric__{i}", mesh, self.settings.material)
    #         bounds = self._scene.scene.bounding_box
    #         self._scene.setup_camera(60, bounds, bounds.get_center())

    def save_object(self):
        description = self.textbox.text_value
        superquadrics = []

        for sq in self.sqs:
            if sq.a1.double_value == 0 or sq.a2.double_value == 0 or sq.a3.double_value == 0:
                continue
            rotation = [[sq.r11.double_value, sq.r12.double_value, sq.r13.double_value],
                        [sq.r21.double_value, sq.r22.double_value, sq.r23.double_value],
                        [sq.r31.double_value, sq.r32.double_value, sq.r33.double_value]]
            scale = [sq.a1.double_value, sq.a2.double_value, sq.a3.double_value]
            position = [sq.x.double_value, sq.y.double_value, sq.z.double_value]
            
            sq_data = {
                "scale": scale,
                "rotation": rotation,
                "position": position,
                "epsilon1": round(sq.e1.double_value, 2),
                "epsilon2": round(sq.e2.double_value, 2)
            }
            superquadrics.append(sq_data)

        new_entry = {
            "description": description,
            "superquadrics": superquadrics
        }

        if os.path.exists(OUTPUT_JSON):
            with open(OUTPUT_JSON, "r") as file:
                data = json.load(file)
        else:
            data = []

        data.append(new_entry)

        with open(OUTPUT_JSON, "w") as file:
            json.dump(data, file, indent=4)

        print("Data saved to", OUTPUT_JSON)
     


    def add_superquadric(self):
        self.sqs.append(Superquadric())
        em = self.window.theme.font_size
        idx = len(self.sqs) - 1
        sq = gui.CollapsableVert(f"SQ{idx+1}", 0.25 * em,
                                gui.Margins(em, 0, 0, 0))
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Shape 1"))
        grid.add_child(self.sqs[-1].e1)
        grid.add_child(gui.Label("Shape 2"))
        grid.add_child(self.sqs[-1].e2)
        grid.add_child(gui.Label("Semi-axis 1"))
        grid.add_child(self.sqs[-1].a1)
        grid.add_child(gui.Label("Semi-axis 2"))
        grid.add_child(self.sqs[-1].a2)
        grid.add_child(gui.Label("Semi-axis 3"))
        grid.add_child(self.sqs[-1].a3)
        grid.add_child(gui.Label("x"))
        grid.add_child(self.sqs[-1].x)
        grid.add_child(gui.Label("y"))
        grid.add_child(self.sqs[-1].y)
        grid.add_child(gui.Label("z"))
        grid.add_child(self.sqs[-1].z)
        grid.add_child(gui.Label("Roll"))
        grid.add_child(self.sqs[-1].roll)
        grid.add_child(gui.Label("Pitch"))
        grid.add_child(self.sqs[-1].pitch)
        grid.add_child(gui.Label("Yaw"))
        grid.add_child(self.sqs[-1].yaw)
        sq.add_child(grid)
        sq.set_is_open(True)
        self.stored_superquadrics.add_child(sq)




    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()
    
    def _on_viz_mode(self, name, index):
        self.viz_mode = name
        #pdb.set_trace()
        if self.viz_mode!="RGB" and self.query_text != "":
            self.update_pcd()
        elif self.viz_mode == "RGB":
            self.update_pcd()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)


    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        self.stop_capture()
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)
        



    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()


    def load(self, path):
        

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        self._scene.scene.clear_geometry()
        if geometry is not None or mesh is not None:
            try:
                if mesh is not None:
                    # Triangle model
                    self._scene.scene.add_model("__model__", mesh)
                else:
                    # Point cloud
                    self._scene.scene.add_geometry("__model__", geometry,
                                                   self.settings.material)
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

def main():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()
    load = True
    w = AppWindow(1024, 768, load)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()