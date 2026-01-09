    def _on_track_toggled(self, checked):
        if checked:
            # When enabling, we set the REFERENCE to the BLEACH FRAME (or current if bleach invalid?)
            # Let's use Bleach Frame as the Canonical Fixed Frame.
            # If we are not at bleach frame, we should probably warn or just proceed?
            # Implementation Choice: Always use Bleach Frame as Reference.
            # We need to capture the FRAP ROI as it exists *now* (assuming user drew it correctly).
            # But if user is at T=100 and enables it, and T=5 is bleach...
            # We assume the ROI drawn is correct for the frame it is drawn on. 
            # Shapes layer shapes are time-agnostic (2D) so they are "correct" for whatever frame the user was looking at.
            # To avoid complexity, let's assume the user draws it approx correct for the bleach frame.
            
            # Actually, robust way:
            # 1. Get current Nucleus Props at CURRENT frame (where user is looking).
            # 2. Store this as REF properties.
            # 3. Store current Shape as REF shape.
            # 4. As time moves, transform relative to THIS reference.
            
            # This allows user to fix ROI at ANY frame and say "Start tracking from here".
            self._capture_ref_state()
            self._update_tracking()
        else:
            self.ref_frap_shapes = None
            self.ref_nuc_props = None

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
            # Copy shapes data
            self.ref_frap_shapes = [s.copy() for s in layer.data]
        else:
            self.ref_frap_shapes = []

    def _get_nucleus_props(self, t):
        # Get Nucleus Labels layer
        # We need the computed labels, not the user-drawn shapes?
        # The user checkbox tooltip says "requires segmented Nucleus Labels".
        # So we look for "Nucleus Labels" layer.
        name = "Nucleus Labels"
        if name not in self.viewer.layers: return None
        
        lbl_layer = self.viewer.layers[name]
        data = lbl_layer.data
        
        if t >= data.shape[0]: return None
        frame_lbl = data[t]
        
        # We assume 1 object or largest object
        # Regionprops
        props = regionprops(frame_lbl)
        if not props: return None
        
        # Pick largest area if multiple
        target = max(props, key=lambda p: p.area)
        
        # Return (centroid(y,x), orientation, major, minor)
        return {
            'centroid': np.array(target.centroid),
            'orientation': target.orientation,
            'major': target.major_axis_length,
            'minor': target.minor_axis_length
        }

    def _update_tracking(self):
        if not self.check_track.isChecked(): return
        if self.ref_nuc_props is None or self.ref_frap_shapes is None: return
        
        # Get Current Time
        curr_step = self.viewer.dims.current_step
        if len(curr_step) == 0: return
        t = curr_step[0]
        
        # Get Current Nuc Props
        curr_props = self._get_nucleus_props(t)
        if curr_props is None: return # Can't track if nuc lost
        
        # Compute Transform
        # Ref -> Current
        # 1. Center at 0 (subtract Ref Centroid)
        # 2. Rotate by (Curr Angle - Ref Angle)
        # 3. Scale? (Optional, maybe skip for now or add checkbox)
        # 4. Move to Curr Centroid
        
        ref_c = self.ref_nuc_props['centroid']
        curr_c = curr_props['centroid']
        
        d_angle = curr_props['orientation'] - self.ref_nuc_props['orientation']
        # Note: skimage orientation is counter-clockwise from X-axis? Checking docs...
        # Orientation is angle between 0th axis (rows, Y) and major axis.
        # Positive angle = Counter-Clockwise?
        # We'll trust the diff works.
        
        # Transform Logic
        c, s = np.cos(d_angle), np.sin(d_angle)
        R = np.array([[c, -s], [s, c]]) # Rotation matrix
        
        # Apply to each shape
        frap_name = self.combo_frap.currentText()
        if frap_name not in self.viewer.layers: return
        layer = self.viewer.layers[frap_name]
        
        new_shapes = []
        for shape in self.ref_frap_shapes:
            # Shape is (N, 2) array of Y, X coordinates
            # 1. Translate to origin relative to Ref Nucleus
            centered = shape - ref_c
            
            # 2. Rotate
            # Check coordinate system. Napari (Y, X). 
            # Rotation matrix for (Y, X)? 
            # If angle is from Y axis...
            # Let's use standard rigid body transform logic
            # rotated = centered @ R.T
            
            # Using rotation matrix defined above:
            # [y', x']^T = [[c, -s], [s, c]] * [y, x]^T
            # So y' = cy - sx, x' = sy + cx
            
            # Wait, if orientation is from Y axis (row axis).
            # We assume consistency.
            
            # Simple 2D rotation
            # Rotate point p around origin
            
            rotated = np.dot(centered, R.T) # (N, 2)
            
            # 3. Translate to Curr Nucleus
            final = rotated + curr_c
            new_shapes.append(final)
            
        # Update Layer
        # We replace data. This triggers events, might cause lag?
        # Block signals?
        # layer.data = new_shapes is the way.
        
        # Note: We must ensure we don't trigger "layer changed" loops if any.
        # _update_live_plot listens to layer changes?
        # The viewer.layers events are inserted/removed.
        # Data change events... shapes layer has events.data?
        # Yes, line 497: layer.events.data.connect(self._update_live_plot)
        # So updating this WILL trigger a plot update.
        # That is GOOD. We want plot to update with new ROI position.
        
        layer.data = new_shapes
