
import numpy as np
import napari
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, 
                            QGroupBox, QFormLayout, QFileDialog, QMessageBox, QCheckBox, QScrollArea, QTabWidget, QApplication)
from qtpy.QtGui import QCursor
from qtpy.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.ndimage import binary_fill_holes
import csv
import os

try:
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class FrapWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Parameters
        self.time_interval = 1.0 
        self.bleach_frame = 5
        
        # Data containers
        self.time_axis = None
        self.norm_curve = None
        self.fit_curve = None
        self.fit_params = None
        
        self.current_save_path = None
        
        # Caching
        self.trace_cache = {} # Map layer_name -> intensity_array
        
        # Tracking State
        self.ref_frap_shapes = None # List of shapes at reference frame
        self.ref_frap_types = None
        self.ref_nuc_props = None   # (centroid, orientation, major, minor) at ref frame
        
        self.setup_ui()
        
        # Connect viewer events
        self.viewer.layers.events.inserted.connect(self._update_layer_combos)
        self.viewer.layers.events.removed.connect(self._update_layer_combos)
        self.viewer.dims.events.current_step.connect(self._on_time_change)
        
        self._update_layer_combos()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.tab_analysis = QWidget()
        self._setup_analysis_tab()
        
        self.tab_segmentation = QWidget()
        self._setup_segmentation_tab()
        
        self.tabs.addTab(self.tab_analysis, "Analysis")
        self.tabs.addTab(self.tab_segmentation, "Segmentation")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def _setup_analysis_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        
        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        
        self.spin_time = QDoubleSpinBox()
        self.spin_time.setValue(1.0)
        self.spin_time.setSingleStep(0.1)
        self.spin_time.valueChanged.connect(self._on_param_change)
        param_layout.addRow("Time Interval (s):", self.spin_time)
        
        self.spin_bleach = QSpinBox()
        self.spin_bleach.setValue(5)
        self.spin_bleach.valueChanged.connect(self._on_param_change)
        param_layout.addRow("Bleach Frame Index:", self.spin_bleach)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # --- Image ---
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Image Layer:"))
        self.combo_image = QComboBox()
        self.combo_image.currentIndexChanged.connect(self._update_live_plot)
        img_layout.addWidget(self.combo_image)
        layout.addLayout(img_layout)

        # --- ROI ---
        roi_group = QGroupBox("ROIs")
        roi_layout = QVBoxLayout()
        
        hbox_roi_btns = QHBoxLayout()
        self.btn_add_rois = QPushButton("Add ROI Layers")
        self.btn_add_rois.clicked.connect(self._add_roi_layers)
        hbox_roi_btns.addWidget(self.btn_add_rois)
        
        self.btn_import_rgn = QPushButton("Import FRAP ROI (.rgn)")
        self.btn_import_rgn.clicked.connect(self._import_rgn_file)
        hbox_roi_btns.addWidget(self.btn_import_rgn)
        
        roi_layout.addLayout(hbox_roi_btns)
        
        form_roi = QFormLayout()
        self.combo_bck = QComboBox()
        self.combo_frap = QComboBox()
        self.combo_nuc = QComboBox()
        
        self.combo_bck.currentIndexChanged.connect(self._update_live_plot)
        self.combo_frap.currentIndexChanged.connect(self._update_live_plot)
        self.combo_nuc.currentIndexChanged.connect(self._update_live_plot)

        form_roi.addRow("Background:", self.combo_bck)
        form_roi.addRow("FRAP:", self.combo_frap)
        form_roi.addRow("Nucleus:", self.combo_nuc)
        
        self.check_track = QCheckBox("Track FRAP ROI with Nucleus")
        self.check_track.setToolTip("Moves the FRAP ROI relative to Nucleus movement (requires segmented Nucleus Labels)")
        self.check_track.toggled.connect(self._on_track_toggled)
        form_roi.addRow(self.check_track)
        
        roi_layout.addLayout(form_roi)
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # --- Fit Settings ---
        fit_group = QGroupBox("Fit Parameters")
        fit_layout = QFormLayout()
        
        self.check_immobile = QCheckBox("Immobile Fraction")
        self.check_immobile.setChecked(True)
        self.check_immobile.toggled.connect(self._update_model_ui)
        fit_layout.addRow(self.check_immobile)
        
        self.combo_components = QComboBox()
        self.combo_components.addItems(["1 Diffusive Population", "2 Diffusive Populations"])
        self.combo_components.setCurrentIndex(1) 
        self.combo_components.currentIndexChanged.connect(self._update_model_ui)
        fit_layout.addRow("Model:", self.combo_components)
        
        def create_bound_row(label, min_val, max_val, default_min, default_max):
            row = QHBoxLayout()
            s_min = QDoubleSpinBox()
            s_min.setMaximum(100000)
            s_min.setValue(default_min)
            s_min.valueChanged.connect(self._update_live_plot)
            
            s_max = QDoubleSpinBox()
            s_max.setMaximum(100000)
            s_max.setValue(default_max)
            s_max.valueChanged.connect(self._update_live_plot)
            
            row.addWidget(QLabel("Min:"))
            row.addWidget(s_min)
            row.addWidget(QLabel("Max:"))
            row.addWidget(s_max)
            
            w = QWidget()
            w.setLayout(row)
            return w, s_min, s_max

        self.wdg_bf, self.spin_bf_min, self.spin_bf_max = create_bound_row("Bound Frac", 0, 100, 0, 100)
        self.lbl_bf = QLabel("Bound Frac (%):")
        fit_layout.addRow(self.lbl_bf, self.wdg_bf)
        
        self.wdg_p1, self.spin_p1_min, self.spin_p1_max = create_bound_row("Pop 1 Frac", 0, 100, 0, 100)
        self.lbl_p1 = QLabel("Pop 1 Frac (%):")
        fit_layout.addRow(self.lbl_p1, self.wdg_p1)
        
        self.wdg_t1, self.spin_t1_min, self.spin_t1_max = create_bound_row("T1", 0, 10000, 0, 1000)
        fit_layout.addRow("T1 (s):", self.wdg_t1)
        
        self.wdg_t2, self.spin_t2_min, self.spin_t2_max = create_bound_row("T2", 0, 10000, 0, 1000)
        self.lbl_t2 = QLabel("T2 (s):")
        fit_layout.addRow(self.lbl_t2, self.wdg_t2)

        fit_group.setLayout(fit_layout)
        layout.addWidget(fit_group)
        
        # --- Results ---
        res_group = QGroupBox("Fit Results")
        res_layout = QVBoxLayout()
        self.lbl_results = QLabel("No Data")
        res_layout.addWidget(self.lbl_results)
        res_group.setLayout(res_layout)
        layout.addWidget(res_group)

        # --- Plotting ---
        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        gs = self.canvas.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        self.ax_main = self.canvas.figure.add_subplot(gs[0])
        self.ax_resid = self.canvas.figure.add_subplot(gs[1], sharex=self.ax_main)
        
        layout.addWidget(self.canvas)
        
        # --- Actions ---
        # --- Actions ---
        btn_layout = QHBoxLayout()
        
        self.btn_save_new = QPushButton("Save New...")
        self.btn_save_new.clicked.connect(self._save_new)
        btn_layout.addWidget(self.btn_save_new)
        
        self.btn_append = QPushButton("Append to:")
        self.btn_append.clicked.connect(self._append)
        btn_layout.addWidget(self.btn_append)
        
        self.lbl_append_path = QLabel("<No File>")
        btn_layout.addWidget(self.lbl_append_path)
        
        self.btn_change_path = QPushButton("Change")
        self.btn_change_path.clicked.connect(self._change_save_path)
        btn_layout.addWidget(self.btn_change_path)
        
        btn_layout.addStretch()

        # Add layout to tab
        lay = QVBoxLayout()
        lay.addWidget(scroll)
        lay.addLayout(btn_layout) # Fix: add buttons to main layout
        self.tab_analysis.setLayout(lay)
        
        self._update_model_ui()

    def _setup_segmentation_tab(self):
        layout = QVBoxLayout()
        
        grp = QGroupBox("Segmentation Parameters")
        form = QFormLayout()
        
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setValue(10.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.valueChanged.connect(self._on_seg_param_change)
        form.addRow("Gaussian Sigma:", self.spin_sigma)
        
        # Morphological Closing
        self.spin_closing = QDoubleSpinBox()
        self.spin_closing.setValue(6.0)
        self.spin_closing.setSingleStep(1.0)
        self.spin_closing.setToolTip("Radius of disk for closing operation")
        self.spin_closing.valueChanged.connect(self._on_seg_param_change)
        form.addRow("Closing Radius:", self.spin_closing)
        
        grp.setLayout(form)
        layout.addWidget(grp)
        
        # Threshold Adjust
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setValue(104)
        self.spin_thresh.setRange(10, 250)
        self.spin_thresh.setSingleStep(1)
        self.spin_thresh.setToolTip("Percentage of Otsu threshold (100 = default, 110 = 1.1x stricter)")
        self.spin_thresh.valueChanged.connect(self._on_seg_param_change)
        form.addRow("Threshold Adjust (%):", self.spin_thresh)

        # Morphological Opening
        self.spin_opening = QDoubleSpinBox()
        self.spin_opening.setValue(0.0)
        self.spin_opening.setSingleStep(1.0)
        self.spin_opening.setToolTip("Radius of disk for opening operation (removes small objects/connections)")
        self.spin_opening.valueChanged.connect(self._on_seg_param_change)
        form.addRow("Opening Radius:", self.spin_opening)
        
        grp.setLayout(form)
        layout.addWidget(grp)
        
        self.check_preview = QCheckBox("Live Preview on Current Frame")
        self.check_preview.toggled.connect(self._on_seg_param_change)
        layout.addWidget(self.check_preview)
        
        btn_seg = QPushButton("Segment Nucleus (All Frames)")
        btn_seg.clicked.connect(self._segment_nucleus_batch)
        layout.addWidget(btn_seg)
        
        layout.addStretch()
        self.tab_segmentation.setLayout(layout)

    def _segment_single_frame(self, frame_data, sigma, closing_radius, thresh_factor, opening_radius):
        if not HAS_SKIMAGE: return None
        
        try:
            blurred = gaussian(frame_data, sigma=sigma)
            thresh = threshold_otsu(blurred) * thresh_factor
            binary = blurred > thresh
            
            if opening_radius > 0:
                selem_open = disk(opening_radius)
                binary = binary_opening(binary, selem_open)
                
            if closing_radius > 0:
                selem_close = disk(closing_radius)
                binary = binary_closing(binary, selem_close)
                
            binary = binary_fill_holes(binary)
            lbl = label(binary)
            if lbl.max() == 0: return np.zeros_like(frame_data, dtype=int)
            
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            largest_label = counts.argmax()
            return (lbl == largest_label).astype(int)
        except:
            return np.zeros_like(frame_data, dtype=int)

    def _on_seg_param_change(self):
        if self.check_preview.isChecked():
            self._update_preview()
        else:
            # Clear preview if unchecked?
            if "Nucleus Preview" in self.viewer.layers:
                self.viewer.layers["Nucleus Preview"].visible = False

    def _on_time_change(self, event=None):
        if self.check_preview.isChecked():
            self._update_preview()

    def _update_preview(self):
        if not HAS_SKIMAGE: return
        
        img_name = self.combo_image.currentText()
        if img_name not in self.viewer.layers: return
        img_layer = self.viewer.layers[img_name]
        data = img_layer.data # T, Y, X
        if data.ndim != 3: return
        
        current_step = self.viewer.dims.current_step
        t = current_step[0]
        if t >= data.shape[0]: return
        
        frame = data[t]
        sigma = self.spin_sigma.value()
        close_radius = self.spin_closing.value()
        thresh_factor = self.spin_thresh.value() / 100.0
        open_radius = self.spin_opening.value()
        
        mask = self._segment_single_frame(frame, sigma, close_radius, thresh_factor, open_radius)
        
        # Show on a Preview Layer
        name = "Nucleus Preview"
        if name not in self.viewer.layers:
            # Create full size empty
            preview_data = np.zeros_like(data, dtype=int)
            self.viewer.add_labels(preview_data, name=name)
        
        layer = self.viewer.layers[name]
        layer.visible = True
        
        # Update only current frame in the 3D array?
        # Directly viewing a 2D result on top of 3D viewer?
        # If I output a 3D array where only t is filled
        # Or I can just update the whole array (slow if big)
        # Or I can make the Preview layer 2D and set its translate to match time?
        # But image is 3D. Napari handles 2D on 3D if dims match?
        # Simplest: Update the slice t of the 3D array.
        
        # To avoid lag, we iterate? No, just set the slice.
        # But layer.data[t] = mask updates the visual?
        # Napari Labels layer refresh might be tricky.
        # Trying direct update.
        
        dtype = layer.data.dtype
        mask = mask.astype(dtype)
        
        # Optimization: Don't re-allocate if shape matches.
        # Just update the slice.
        # However, we should probably clear other slices to avoid confusion if user scrolls?
        # "Live Preview on Current Frame" implies it updates as you scroll.
        # So as you scroll, `_on_time_change` calls this, and paints the new frame.
        # The previous frames will stick around unless we clear them.
        # For preview, maybe we don't care if old previews persist?
        # Ideally, we want to see ONLY the current one.
        # So we might want to clear the preview layer or use a 2D layer.
        
        # Let's try 2D layer approach for speed and clarity.
        # 2D layer on top of 3D image.
        # We assume user is viewing 2D slice.
        # Update data to 2D mask.
        # But we need to ensure it's visible.
        
        if layer.data.ndim == 3:
            # Recreate as 2D?
            self.viewer.layers.remove(name)
            self.viewer.add_labels(mask, name=name)
        else:
            layer.data = mask
            
        # We don't need to translate if it's 2D and we just want to see it?
        # But 3D viewer dims...
        # If image is 3D, viewer slider T controls which slice of image is shown.
        # If 2D layer is added, it typically shows on all T.
        # This is exactly what we want for "Live Preview": show the mask derived from CURRENT frame on CURRENT view.
        # Perfect.
        
    def _segment_nucleus_batch(self):
        if not HAS_SKIMAGE: return

        img_name = self.combo_image.currentText()
        if img_name not in self.viewer.layers: return
        img_layer = self.viewer.layers[img_name]
        data = img_layer.data
        if data.ndim != 3: return
            
            
        sigma = self.spin_sigma.value()
        close_radius = self.spin_closing.value()
        thresh_factor = self.spin_thresh.value() / 100.0
        open_radius = self.spin_opening.value()
        
        labels_out = np.zeros_like(data, dtype=int)
        self.viewer.status = "Segmenting..."
        
        try:
            # We can use logic similar to preview but loop
            # Maybe use napari progress
            
            for t in range(data.shape[0]):
                mask = self._segment_single_frame(data[t], sigma, close_radius, thresh_factor, open_radius)
                labels_out[t] = mask
        
            name = "Nucleus Labels"
            if name in self.viewer.layers:
                self.viewer.layers[name].data = labels_out
                self.viewer.layers[name].visible = True
            else:
                self.viewer.add_labels(labels_out, name=name)
            
            # Hide Preview
            self.check_preview.setChecked(False)  # Uncheck box to stop live updates
            if "Nucleus Preview" in self.viewer.layers:
                self.viewer.layers.remove("Nucleus Preview")
                 
            self.viewer.status = "Segmentation Complete"
            QMessageBox.information(self, "Success", "Nucleus segmentation complete.")
            
            self.tabs.setCurrentIndex(0)
            self._update_live_plot()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Segmentation failed: {str(e)}")

    def _update_model_ui(self):
        has_immobile = self.check_immobile.isChecked()
        is_2_comp = self.combo_components.currentIndex() == 1
        
        self.lbl_bf.setVisible(has_immobile)
        self.wdg_bf.setVisible(has_immobile)
        self.lbl_p1.setVisible(is_2_comp)
        self.wdg_p1.setVisible(is_2_comp)
        self.lbl_t2.setVisible(is_2_comp)
        self.wdg_t2.setVisible(is_2_comp)
        
        self._update_live_plot()

    def _on_param_change(self):
        self.time_interval = self.spin_time.value()
        self.bleach_frame = self.spin_bleach.value()
        self._update_live_plot()

    def _update_layer_combos(self, event=None):
        def update_combo(combo, layer_type, target_name=None):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            layers = [l for l in self.viewer.layers if isinstance(l, layer_type)]
            items = [l.name for l in layers]
            combo.addItems(items)
            idx = -1
            if current in items: idx = items.index(current)
            elif target_name and target_name in items: idx = items.index(target_name)
            elif items: idx = 0
            if idx >= 0: combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        
        update_combo(self.combo_image, napari.layers.Image)
        update_combo(self.combo_bck, napari.layers.Shapes, "Background")
        update_combo(self.combo_frap, napari.layers.Shapes, "FRAP")
        update_combo(self.combo_nuc, napari.layers.Shapes, "Nucleus")
        self._update_live_plot()

    def _add_roi_layers(self):
        def ensure_layer(name, color):
            if name not in self.viewer.layers:
                layer = self.viewer.add_shapes(
                    name=name,
                    edge_color=color,
                    face_color=color,
                    opacity=0.3,
                    ndim=2
                )
                layer.events.data.connect(self._on_roi_data_change)
            else:
                self.viewer.layers[name].opacity = 0.3
        
        ensure_layer("Nucleus", "green")
        ensure_layer("FRAP", "red")
        ensure_layer("Background", "blue")
        
        self._update_layer_combos()
        
        def set_combo(combo, name):
            idx = combo.findText(name)
            if idx >= 0: combo.setCurrentIndex(idx)
            
        set_combo(self.combo_bck, "Background")
        set_combo(self.combo_frap, "FRAP")
        set_combo(self.combo_nuc, "Nucleus")

    def _import_rgn_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Metamorph RGN", "", "Region Files (*.rgn);;All Files (*)")
        if not path: return
        
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                tags = {}
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    sub = p.split()
                    if not sub: continue
                    tag_id = int(sub[0])
                    tags[tag_id] = [int(x) for x in sub[1:]]
                
                if 0 not in tags: continue
                r_type = tags[0][0]
                if 2 not in tags: continue
                x_pos, y_pos = tags[2]
                if 6 not in tags: continue
                data_pts = tags[6]
                
                shape_data = None
                shape_type = 'polygon'
                
                if r_type == 1 and len(data_pts) >= 3 and data_pts[0] == 2:
                     w, h = data_pts[1], data_pts[2]
                     shape_data = np.array([
                         [y_pos, x_pos],
                         [y_pos+h, x_pos],
                         [y_pos+h, x_pos+w],
                         [y_pos, x_pos+w]
                     ])
                     shape_type = 'rectangle'
                elif len(data_pts) > 3 and data_pts[0] > 2:
                     coords = np.array(data_pts[1:]).reshape(-1, 2)
                     shape_data = coords[:, ::-1] 
                     shape_type = 'polygon'
                
                frap_layer_name = self.combo_frap.currentText()
                if frap_layer_name not in self.viewer.layers:
                    self._add_roi_layers()
                    frap_layer_name = "FRAP"
                
                layer = self.viewer.layers[frap_layer_name]
                layer.add(shape_data, shape_type=shape_type)
                
            QMessageBox.information(self, "Success", f"Imported ROI(s) from {os.path.basename(path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", f"Could not parse RGN.\n{str(e)}")

    def _on_roi_data_change(self, event):
        layer = event.source
        if layer.name in self.trace_cache:
            del self.trace_cache[layer.name]
        self._update_live_plot()

    def _on_track_toggled(self, checked):
        if checked:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.check_track.repaint()
            QApplication.processEvents()
            
            try:
                self._capture_ref_state()
                self._compute_tracked_rois_batch()
                self._update_live_plot()
            finally:
                QApplication.restoreOverrideCursor()
        else:
            self._restore_tracking()
            self.ref_frap_shapes = None
            self.ref_frap_types = None
            self.ref_nuc_props = None
            self._update_live_plot()

    def _restore_tracking(self):
        # Restore original shapes if available
        if self.ref_frap_shapes is None: return
        
        frap_name = self.combo_frap.currentText()
        if frap_name not in self.viewer.layers: return 
        layer = self.viewer.layers[frap_name]
        
        # In-place update to avoid VisPy crash
        # Use ref types if available, else fallback
        types = self.ref_frap_types if self.ref_frap_types else ['polygon']*len(self.ref_frap_shapes)
        
        if len(types) < len(self.ref_frap_shapes):
            types = types + ['polygon'] * (len(self.ref_frap_shapes) - len(types))
            
        layer.data = self.ref_frap_shapes
        layer.shape_type = types
            
        # No need to restore combo or reconnect events since layer object persists

    def _compute_tracked_rois_batch(self):
        if not self.check_track.isChecked(): return
        if self.ref_nuc_props is None or self.ref_frap_shapes is None: return
        
        frap_name = self.combo_frap.currentText()
        if frap_name not in self.viewer.layers: return
        layer = self.viewer.layers[frap_name]
        
        img_name = self.combo_image.currentText()
        if img_name not in self.viewer.layers: return
        img_data = self.viewer.layers[img_name].data
        if img_data.ndim != 3: return
        n_frames = img_data.shape[0]
        
        ref_c = self.ref_nuc_props['centroid']
        ref_angle = self.ref_nuc_props['orientation']
        
        all_new_shapes = []
        all_new_types = []
        
        self.viewer.status = "Tracking FRAP ROI..."
        
        for t in range(n_frames):
            curr_props = self._get_nucleus_props(t)
            if curr_props is None: continue
                
            curr_c = curr_props['centroid']
            d_angle = curr_props['orientation'] - ref_angle
            
            c, s = np.cos(d_angle), np.sin(d_angle)
            R = np.array([[c, -s], [s, c]])
            
            for i, shape in enumerate(self.ref_frap_shapes):
                centered = shape - ref_c
                rotated = np.dot(centered, R.T)
                final_2d = rotated + curr_c
                
                t_col = np.full((final_2d.shape[0], 1), t)
                final_3d = np.hstack([t_col, final_2d])
                all_new_shapes.append(final_3d)
                
                if self.ref_frap_types and i < len(self.ref_frap_types):
                    all_new_types.append(self.ref_frap_types[i])
                else:
                    all_new_types.append('polygon') 
        
        if all_new_shapes:
            # In-place update to avoid VisPy crash
            layer.data = all_new_shapes
            # shape_type update might be needed if types mixed, but for tracking usually uniform or same order
            if len(all_new_types) == len(all_new_shapes):
                 layer.shape_type = all_new_types
        
        self.viewer.status = "Tracking Complete"
        QMessageBox.information(self, "Tracking Complete", "FRAP ROIs have been tracked/updated for all frames.")

    def _capture_ref_state(self):
        # 1. Get Current Time
        curr_step = self.viewer.dims.current_step
        if len(curr_step) > 0:
            t = curr_step[0]
        else:
            return

        # 2. Get Nucleus Props at T
        props = self._get_nucleus_props(t)
        if props is None:
            QMessageBox.warning(self, "Tracking Error", "Could not find a single nucleus at current frame to track.\nEnsure 'Nucleus Labels' layer exists and has 1 object.")
            self.check_track.setChecked(False)
            return
            
        self.ref_nuc_props = props
        
        # 3. Get FRAP Shapes
        frap_name = self.combo_frap.currentText()
        if frap_name in self.viewer.layers:
            layer = self.viewer.layers[frap_name]
            # Copy shapes data and types
            self.ref_frap_shapes = [s.copy() for s in layer.data]
            self.ref_frap_types = list(layer.shape_type)
        else:
            self.ref_frap_shapes = []
            self.ref_frap_types = []

    def _get_nucleus_props(self, t):
        name = "Nucleus Labels"
        if name not in self.viewer.layers: return None
        lbl_layer = self.viewer.layers[name]
        data = lbl_layer.data
        if t >= data.shape[0]: return None
        frame_lbl = data[t]
        props = regionprops(frame_lbl)
        if not props: return None
        target = max(props, key=lambda p: p.area)
        return {
            'centroid': np.array(target.centroid),
            'orientation': target.orientation,
            'major': target.major_axis_length,
            'minor': target.minor_axis_length
        }

    # Removed _update_tracking (live)

    def _get_layer_by_name(self, name):
        if not name: return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _compute_data(self):
        img_name = self.combo_image.currentText()
        bck_name = self.combo_bck.currentText()
        frap_name = self.combo_frap.currentText()
        nuc_name = self.combo_nuc.currentText()
        
        img_layer = self._get_layer_by_name(img_name)
        bck_layer = self._get_layer_by_name(bck_name)
        frap_layer = self._get_layer_by_name(frap_name)
        nuc_layer = self._get_layer_by_name(nuc_name)
        
        if not (img_layer and bck_layer and frap_layer and nuc_layer):
            return None, None
            
        data = img_layer.data 
        if data.ndim != 3: return None, None
            
        def get_mean(layer):
            if layer is None: return np.zeros(data.shape[0])
            
            # Check Cache
            if layer.name in self.trace_cache:
                cached = self.trace_cache[layer.name]
                if len(cached) == data.shape[0]:
                    return cached
            
            if len(layer.data) == 0: 
                res = np.zeros(data.shape[0])
                self.trace_cache[layer.name] = res
                return res
            
            try:
                # Decide mask dimensionality based on layer dimensionality
                if layer.ndim == 2:
                    # 2D layer: get 2D mask (Y, X)
                    mask_2d = layer.to_masks(mask_shape=data.shape[1:]).sum(axis=0) > 0
                    if mask_2d.sum() == 0: 
                        res = np.zeros(data.shape[0])
                        self.trace_cache[layer.name] = res
                        return res
                    
                    # Apply 2D mask to each frame of 3D data
                    res = data[:, mask_2d].mean(axis=1)
                    self.trace_cache[layer.name] = res
                    return res
                    
                else: 
                    # 3D layer
                    mask_3d = layer.to_masks(mask_shape=data.shape).sum(axis=0) > 0
                    
                    # Weighted mean per frame
                    sums = (data * mask_3d).sum(axis=(1, 2))
                    counts = mask_3d.sum(axis=(1, 2))
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        means = sums / counts
                    res = np.nan_to_num(means)
                    self.trace_cache[layer.name] = res
                    return res

            except Exception as e:
                print(f"Mask generation failed: {e}")
                return np.zeros(data.shape[0])

        i_bck = get_mean(bck_layer)
        i_frap = get_mean(frap_layer)
        
        labels_name = "Nucleus Labels"
        i_nuc = None
        
        # For Nucleus Labels, we also cache?
        # Labels layer might change less often, but let's cache it too.
        # But labels data is the mask itself. 
        # Checking if labels name in cache requires us to invalidate it when labels change.
        # We haven't connected listener for Labels layer yet.
        # For safety, let's leave Nucleus Labels uncached or handle it carefully.
        # The user's main lag is from FRAP ROI (Shapes) which is complex. Labels is integer lookup.
        
        if labels_name in self.viewer.layers:
            # Simple check if current data matches cached
            pass # Keep logic as is for labels for now to minimize risk
            
        if labels_name in self.viewer.layers:
            lbl_layer = self.viewer.layers[labels_name]
            if lbl_layer.data.shape == data.shape:
                labels = lbl_layer.data
                means = []
                # This could be slow too. 
                # Let's consider caching i_nuc if we can key it to something unique?
                # Or just assume if not 'dirty', it's good.
                # But we don't track labels dirtiness.
                
                # Optimized extraction for labels?
                # For now, let's stick to existing logic for labels, focus on Shapes.
                for t in range(data.shape[0]):
                    mask = labels[t] > 0
                    if mask.sum() > 0:
                        means.append(data[t][mask].mean())
                    else:
                        means.append(0)
                i_nuc = np.array(means)
        
        if i_nuc is None:
            # Fallback to explicit Nucleus ROI layer
            i_nuc = get_mean(nuc_layer)
        
        i_frap_sub = np.maximum(0, i_frap - i_bck)
        i_nuc_sub = np.maximum(0, i_nuc - i_bck)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            valid = i_nuc_sub > 0
            i_corr = np.zeros_like(i_nuc_sub)
            i_corr[valid] = i_frap_sub[valid] / i_nuc_sub[valid]
            
            if self.bleach_frame < len(i_corr):
                min_val = i_corr[self.bleach_frame]
            else:
                min_val = np.min(i_corr) if len(i_corr) > 0 else 0
            i_shifted = i_corr - min_val
            
            pre_val = np.mean(i_shifted[:self.bleach_frame]) if self.bleach_frame > 0 else i_shifted[0]
            if pre_val == 0 or np.isnan(pre_val):
                i_norm = i_shifted 
            else:
                i_norm = i_shifted / pre_val
                
        i_norm = np.nan_to_num(i_norm)
        timestamps = np.arange(len(i_norm)) * self.time_interval
        
        self.time_axis = timestamps
        self.norm_curve = i_norm
        return timestamps, i_norm

    def _update_live_plot(self, event=None):
        t, y = self._compute_data()
        self.ax_main.clear()
        self.ax_resid.clear()
        
        if t is not None:
            self.ax_main.plot(t, y, label="Data", color='blue')
            bleach_t = self.bleach_frame * self.time_interval
            self.ax_main.axvline(bleach_t, color='gray', linestyle='--')
            self.ax_main.axhline(1.0, color='black', linestyle=':', label="Ref (1.0)")
            
            self._perform_fit()
            
            if self.fit_curve is not None:
                start_idx = self.bleach_frame
                if start_idx < len(t):
                    t_fit = t[start_idx:]
                    y_fit = self.fit_curve
                    y_data = y[start_idx:]
                    self.ax_main.plot(t_fit, y_fit, 'r--', label="Fit")
                    residuals = y_data - y_fit
                    self.ax_resid.plot(t_fit, residuals, color='red')
                    self.ax_resid.axhline(0, color='black', linestyle=':')
            
            self.ax_main.legend()
            self.ax_main.set_ylabel("Intensity")
            self.ax_resid.set_ylabel("Resid.")
            self.ax_resid.set_xlabel("Time (s)")
            
        self.canvas.draw()

    # Models
    def _model_1_diff_immobile(self, t, a, t1):
        return a * (1 - np.exp(-t / t1))
    
    def _model_1_diff_no_immobile(self, t, t1):
        return 1.0 * (1 - np.exp(-t / t1))
        
    def _model_2_diff_immobile(self, t, a, b, t1, t2):
        return a * (1 - b * np.exp(-t / t1) - (1 - b) * np.exp(-t / t2))
        
    def _model_2_diff_no_immobile(self, t, b, t1, t2):
        return 1.0 * (1 - b * np.exp(-t / t1) - (1 - b) * np.exp(-t / t2))

    def _perform_fit(self):
        if self.norm_curve is None:
            self.lbl_results.setText("No Data")
            return
            
        start_idx = self.bleach_frame
        if start_idx >= len(self.norm_curve):
            return
            
        y_data = self.norm_curve[start_idx:]
        t_data = self.time_axis[start_idx:] - self.time_axis[start_idx]
        
        has_immobile = self.check_immobile.isChecked()
        is_2_comp = self.combo_components.currentIndex() == 1
        
        bf_min = self.spin_bf_min.value()
        bf_max = self.spin_bf_max.value()
        p1_min = self.spin_p1_min.value()
        p1_max = self.spin_p1_max.value()
        t1_min = self.spin_t1_min.value()
        t1_max = self.spin_t1_max.value()
        t2_min = self.spin_t2_min.value()
        t2_max = self.spin_t2_max.value()
        
        a_min = max(0, 1.0 - (bf_max / 100.0))
        a_max = min(1.5, 1.0 - (bf_min / 100.0))
        if a_min > a_max: a_min, a_max = a_max, a_min
        
        b_min = p1_min / 100.0
        b_max = p1_max / 100.0
        
        try:
            popt = None
            res_str = ""
            
            if not is_2_comp:
                if has_immobile:
                    p0 = [(a_min+a_max)/2, (t1_min+t1_max)/2]
                    bounds = ([a_min, t1_min], [a_max, t1_max])
                    popt, _ = curve_fit(self._model_1_diff_immobile, t_data, y_data, p0=p0, bounds=bounds)
                    self.fit_curve = self._model_1_diff_immobile(t_data, *popt)
                    
                    bf_val = (1 - popt[0]) * 100
                    res_str = f"Bound Frac: {bf_val:.1f}%\nt1: {popt[1]:.3f}s"
                    self.fit_params = ["1_Diff_Immobile", popt[0], popt[1]]
                else:
                    p0 = [(t1_min+t1_max)/2]
                    bounds = ([t1_min], [t1_max])
                    popt, _ = curve_fit(self._model_1_diff_no_immobile, t_data, y_data, p0=p0, bounds=bounds)
                    self.fit_curve = self._model_1_diff_no_immobile(t_data, *popt)
                    res_str = f"Bound Frac: 0% (Fixed)\nt1: {popt[0]:.3f}s"
                    self.fit_params = ["1_Diff_Fixed", 1.0, popt[0]]
            else:
                if has_immobile:
                    p0 = [(a_min+a_max)/2, (b_min+b_max)/2, 10.0, 50.0]
                    bounds = ([a_min, b_min, t1_min, t2_min], [a_max, b_max, t1_max, t2_max])
                    popt, _ = curve_fit(self._model_2_diff_immobile, t_data, y_data, p0=p0, bounds=bounds)
                    self.fit_curve = self._model_2_diff_immobile(t_data, *popt)
                    
                    bf_val = (1 - popt[0]) * 100
                    p1_val = popt[1] * 100
                    p2_val = (1 - popt[1]) * 100
                    res_str = f"Bound: {bf_val:.1f}%\nPop1: {p1_val:.1f}%\nPop2: {p2_val:.1f}%\nt1: {popt[2]:.3f}s\nt2: {popt[3]:.3f}s"
                    self.fit_params = ["2_Diff_Immobile", *popt]
                else:
                    p0 = [(b_min+b_max)/2, 10.0, 50.0]
                    bounds = ([b_min, t1_min, t2_min], [b_max, t1_max, t2_max])
                    popt, _ = curve_fit(self._model_2_diff_no_immobile, t_data, y_data, p0=p0, bounds=bounds)
                    self.fit_curve = self._model_2_diff_no_immobile(t_data, *popt)
                    
                    p1_val = popt[0] * 100
                    p2_val = (1 - popt[0]) * 100
                    res_str = f"Bound: 0%\nPop1: {p1_val:.1f}%\nPop2: {p2_val:.1f}%\nt1: {popt[1]:.3f}s\nt2: {popt[2]:.3f}s"
                    self.fit_params = ["2_Diff_Fixed", 1.0, *popt]
            
            self.lbl_results.setText(res_str)
            
        except Exception as e:
            self.lbl_results.setText(f"Fit Failed")
            self.fit_curve = None
            self.fit_params = None

    def _save_new(self):
        if self.norm_curve is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save New Results", "", "CSV Files (*.csv)")
        if not path: return
        
        # Ensure it ends in .csv? logic handles it via splitext usually but let's be safe
        if not path.lower().endswith('.csv'):
            path += ".csv"
            
        self.current_save_path = path
        self._write_data(path)
        self._update_append_label()
        
    def _append(self):
        if self.norm_curve is None: return
        
        path = self.current_save_path
        if not (path and os.path.exists(path)):
            path, _ = QFileDialog.getOpenFileName(self, "Append to Results", "", "CSV Files (*.csv)")
            if not path: return
            
        self._write_data(path)

    def _change_save_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Result File", "", "CSV Files (*.csv)")
        if not path: return
        self.current_save_path = self._resolve_params_path(path)
        self._update_append_label()

    def _update_append_label(self):
        if self.current_save_path:
            self.lbl_append_path.setText(os.path.basename(self.current_save_path))
        else:
            self.lbl_append_path.setText("<No File>")

    def _resolve_params_path(self, path):
        # If it's already a params file, return it
        if path.endswith("_params.csv"):
            return path
            
        # If it's a curves file, try to find the sibling params file
        if path.endswith("_curves.csv"):
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            
            # Simple heuristic: Look for any *_params.csv in the folder
            # If the curves file starts with the params basename, match it.
            try:
                candidates = [f for f in os.listdir(dirname) if f.endswith("_params.csv")]
                for cand in candidates:
                    cand_base = cand.replace("_params.csv", "")
                    if basename.startswith(cand_base):
                        return os.path.join(dirname, cand)
            except:
                pass
                
        # Fallback: assume the user selected a base name or generic file
        return path

    def _write_data(self, path):
        # 1. Resolve the actual Params file path
        param_file = self._resolve_params_path(path)
        
        # Update current path
        self.current_save_path = param_file
        self._update_append_label()
        
        base, _ = os.path.splitext(param_file)
        if base.endswith("_params"):
            base = base[:-7]
            
        curves_file = base + "_curves.csv"
        img_name = self.combo_image.currentText()
        
        try:
            # --- Write Curves (Update/Append to single file) ---
            curves_header = ["Filename", "Time (s)", "Normalized Intensity", "Fitted Value"]
            curves_rows = []
            
            # Read existing if available
            if os.path.exists(curves_file) and os.path.getsize(curves_file) > 0:
                with open(curves_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    all_rows = list(reader)
                    if all_rows:
                        # Robust read: Assume first row is header, keep others
                        # Check if it looks like a valid file (at least 1 row with correct cols or just header)
                        # We won't strictly check header text to avoid "Time(s)" vs "Time (s)" issues
                        for row in all_rows[1:]:
                            if row and len(row) >= 4 and row[0] != img_name:
                                curves_rows.append(row)
            
            # Prepare new curve data
            fit_vals = np.full(len(self.norm_curve), np.nan)
            if self.fit_curve is not None:
                start_idx = self.bleach_frame
                fit_vals[start_idx:] = self.fit_curve
            
            for t, y, yf in zip(self.time_axis, self.norm_curve, fit_vals):
                curves_rows.append([img_name, t, y, yf])
                
            # Write Curves
            with open(curves_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(curves_header)
                writer.writerows(curves_rows)
            
            # --- Write Params (Update/Append) ---
            rows = []
            header = ["Filename", "Model", "a (Plateau)", "b (Pop1)", "t1", "t2"]
            if os.path.exists(param_file) and os.path.getsize(param_file) > 0:
                with open(param_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    file_rows = list(reader)
                    if file_rows:
                        # Robust read: keep existing rows if they look valid
                        # We always use our standard header for writing
                        # We ignore the existing header comparison
                        for row in file_rows[1:]:
                             if row and len(row) >= 6:
                                 rows.append(row)
            
            new_row = [img_name, "No Fit", 0, 0, 0, 0]
            if self.fit_params is not None:
                model_name = self.fit_params[0]
                a, b, t1, t2 = 0, 0, 0, 0
                vals = self.fit_params[1:]
                if "1_Diff" in model_name:
                    a = vals[0]
                    t1 = vals[1]
                elif "2_Diff" in model_name:
                    if "Fixed" in model_name:
                         a, b, t1, t2 = 1.0, vals[0], vals[1], vals[2]
                    else:
                         a, b, t1, t2 = vals
                new_row = [img_name, model_name, a, b, t1, t2]
            
            updated = False
            for i, row in enumerate(rows):
                if row and row[0] == img_name:
                    rows[i] = new_row
                    updated = True
                    break
            
            if not updated:
                rows.append(new_row)
            
            with open(param_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            QMessageBox.information(self, "Saved", f"Results updated for {img_name}.\nFiles: {os.path.basename(param_file)} & {os.path.basename(curves_file)}")

        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save files.\n{str(e)}")
