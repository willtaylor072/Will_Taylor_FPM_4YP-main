"""
This code is based on the picamera2 example code from the picamera2 repository and has been modified by the author:
Author: Álvaro Fernández Galiana
Email: afernandezgaliana@schmidtsciencefellows.org

The original code can be found at: https://github.com/raspberrypi/picamera2

The original code was under the following license:

BSD 2-Clause License

Copyright (c) 2021, Raspberry Pi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPalette
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSlider, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput, FileOutput
from picamera2.previews.qt import QGlPicamera2

try:
    import cv2
    cv_present = True
except ImportError:
    cv_present = False
    print("OpenCV not found - HDR not available")
import threading
import numpy as np

class RPiCameraApp:
    def __init__(self):
        self.implemented_controls = [
            "ColourCorrectionMatrix",
            "Saturation",
            "Contrast",
            "Sharpness",
            "Brightness",
            "NoiseReductionMode",
            "AeEnable",
            "AeMeteringMode",
            "AeConstraintMode",
            "AeExposureMode",
            "AwbEnable",
            "AwbMode",
            "ExposureValue",
            "ExposureTime",
            "AnalogueGain",
            "ColourGains",
            "ScalerCrop",
            "FrameDurationLimits"
        ]

        self.ignore_controls = {
            "AfMode",
            "AfTrigger",
            "AfSpeed",
            "AfRange",
            "AfWindows",
            "AfPause",
            "AfMetering",
            "ScalerCrops"
        }
        self.picam2 = Picamera2()
        self.scaler_crop = None
        self.recording = False
        self.hdr_imgs = {"exposures": None}
        self.qpicamera2 = None
        self.lores_size = None  # Initialize lores_size
        self.still_kwargs = None  # Initialize still_kwargs
        self.configure_camera()
        self.implemented_controls = self.implemented_controls
        self.ignore_controls = self.ignore_controls

    def configure_camera(self):
        self.picam2.post_callback = self.post_callback
        self.lores_size = self.picam2.sensor_resolution
        while self.lores_size[0] > 1600:
            self.lores_size = (self.lores_size[0] // 2 & ~1, self.lores_size[1] // 2 & ~1)
        self.still_kwargs = {"lores": {"size": self.lores_size}, "display": "lores", "encode": "lores", "buffer_count": 1}
        self.picam2.still_configuration = self.picam2.create_still_configuration(
            **self.still_kwargs,
        )
        self.picam2.configure("still")
        _ = self.picam2.sensor_modes
        
        # Initialize scaler_crop
        _, self.scaler_crop, _ = self.picam2.camera_controls['ScalerCrop']

    def post_callback(self, request):
        metadata = request.get_metadata()
        sorted_metadata = sorted(metadata.items(), key=lambda x: x[0] if "Awb" not in x[0] else f"Z{x[0]}")
        pretty_metadata = []
        for k, v in sorted_metadata:
            row = ""
            try:
                iter(v)
                if k == "ColourCorrectionMatrix":
                    matrix = np.around(np.reshape(v, (-1, 3)), decimals=2)
                    row = f"{k}:\n{matrix}"
                else:
                    row_data = [f'{x:.2f}' if type(x) is float else f'{x}' for x in v]
                    row = f"{k}: ({', '.join(row_data)})"
            except TypeError:
                if type(v) is float:
                    row = f"{k}: {v:.2f}"
                else:
                    row = f"{k}: {v}"
            pretty_metadata.append(row)
        self.info_tab.setText('\n'.join(pretty_metadata))

        if not self.aec_tab.exposure_time.isEnabled():
            self.aec_tab.exposure_time.setValue(metadata["ExposureTime"])
            self.aec_tab.analogue_gain.setValue(metadata["AnalogueGain"])
        if not self.aec_tab.colour_gain_r.isEnabled():
            self.aec_tab.colour_gain_r.setValue(metadata.get("ColourGains", [1.0, 1.0])[0])
            self.aec_tab.colour_gain_b.setValue(metadata.get("ColourGains", [1.0, 1.0])[1])
        self.vid_tab.frametime = metadata["FrameDuration"]


    
    def update_controls(self):
        # Ensure scaler_crop is accessed via self
        scaler_crop = self.scaler_crop
        
        _, full_img, _ = self.picam2.camera_controls['ScalerCrop']
        ar = full_img[2] / full_img[3]
        new_scaler_crop = list(scaler_crop)
        new_scaler_crop[3] = int(new_scaler_crop[2] / ar)
        new_scaler_crop[1] += (scaler_crop[3] - new_scaler_crop[3]) // 2

        new_scaler_crop[1] = max(new_scaler_crop[1], full_img[1])
        new_scaler_crop[1] = min(new_scaler_crop[1], full_img[1] + full_img[3] - new_scaler_crop[3])
        new_scaler_crop[0] = max(new_scaler_crop[0], full_img[0])
        new_scaler_crop[0] = min(new_scaler_crop[0], full_img[0] + full_img[2] - new_scaler_crop[2])

        self.scaler_crop = tuple(new_scaler_crop)

        with self.picam2.controls as controls:
            controls.ScalerCrop = self.scaler_crop
        self.aec_tab.aec_update()
        self.aec_tab.awb_update()
        self.vid_tab.vid_update()
        self.pic_tab.pic_update()

        self.pan_tab.pan_display.update()

        self.vid_tab.resolution_h.setValue(self.picam2.video_configuration.main.size[1])
        self.vid_tab.resolution_w.setValue(self.picam2.video_configuration.main.size[0])
        self.pic_tab.resolution_h.setValue(self.picam2.still_configuration.main.size[1])
        self.pic_tab.resolution_w.setValue(self.picam2.still_configuration.main.size[0])

    def switch_config(self, new_config):
        print("Switching to", new_config)
        self.picam2.stop()
        self.picam2.configure(new_config)
        self.update_controls()
        self.picam2.start()
        self.update_controls()

    def on_rec_button_clicked(self):
        if self.mode_tabs.currentIndex():
            self.on_vid_button_clicked()
        else:
            self.on_pic_button_clicked()

    def on_vid_button_clicked(self):
        if not self.recording:
            self.mode_tabs.setEnabled(False)
            encoder = H264Encoder()
            if self.vid_tab.filetype.currentText() in ["mp4", "mkv", "mov", "ts", "avi"]:
                output = FfmpegOutput(
                    f"{self.vid_tab.filename.text() if self.vid_tab.filename.text() else 'test'}.{self.vid_tab.filetype.currentText()}"
                )
            else:
                output = FileOutput(
                    f"{self.vid_tab.filename.text() if self.vid_tab.filename.text() else 'test'}.{self.vid_tab.filetype.currentText()}"
                )
            self.picam2.start_encoder(encoder, output, self.vid_tab.quality)
            self.rec_button.setText("Stop recording")
            self.recording = True
        else:
            self.picam2.stop_encoder()
            self.rec_button.setText("Start recording")
            self.mode_tabs.setEnabled(True)
            self.recording = False

    def on_pic_button_clicked(self):
        if self.pic_tab.preview_check.isChecked() and self.rec_button.isEnabled():
            self.switch_config("still")
            self.picam2.capture_request(signal_function=self.qpicamera2.signal_done)
        else:
            self.picam2.capture_request(signal_function=self.qpicamera2.signal_done)
        self.rec_button.setEnabled(False)
        self.mode_tabs.setEnabled(False)

    def on_mode_change(self, i):
        if self.recording:
            print("Not switching, recording in progress, so back to video")
            self.mode_tabs.setCurrentIndex(1)
            return
        print(f"Switch to {'video' if i else 'photo'}")
        self.vid_tab.reset()
        self.pic_tab.reset()
        if i:
            self.rec_button.setText("Start recording")
            self.switch_config("video")
            self.hide_button.setEnabled(True)
        else:
            self.rec_button.setText("Take photo")
            self.switch_config("preview" if self.pic_tab.preview_check.isChecked() else "still")
            self.pic_tab.apply_settings()
            
    def capture_done(self, job):
        if not self.pic_tab.hdr.isChecked():
            request = self.picam2.wait(job)
            if self.pic_tab.filetype.currentText() == "raw":
                request.save_dng(
                    f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}.dng"
                )
            else:
                request.save(
                    "main", f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}.{self.pic_tab.filetype.currentText()}"
                )
            request.release()
            self.rec_button.setEnabled(True)
            self.mode_tabs.setEnabled(True)
            if self.pic_tab.preview_check.isChecked():
                self.switch_config("preview")
        else:
            request = self.picam2.wait(job)
            new_img = request.make_array("main")
            new_cv_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            metadata = request.get_metadata()
            request.release()
            new_exposure = metadata["ExposureTime"]
            if self.hdr_imgs["exposures"] is None:
                self.pic_tab.pic_update()
                e_log = np.log2(new_exposure)
                max_e = np.log2(self.pic_tab.pic_dict["FrameDurationLimits"][1])
                below = self.pic_tab.stops_hdr_below.value()
                above = self.pic_tab.stops_hdr_above.value()
                if e_log + 1 > max_e:
                    above = max_e - e_log
                    print("Desired exposure too long, reducing", e_log + 1, max_e, above)
                self.hdr_imgs["exposures"] = {"all": list(set(np.logspace(
                    e_log - below, e_log + above, self.pic_tab.num_hdr.value(),
                    base=2.0, dtype=np.integer
                )))}
                if 0 in self.hdr_imgs["exposures"]["all"]:
                    i = self.hdr_imgs["exposures"]["all"].index(0)
                    self.hdr_imgs["exposures"]["all"][i] = self.picam2.camera_controls["ExposureTime"][0]
                self.hdr_imgs["exposures"]["all"].sort()
                self.hdr_imgs["exposures"]["left"] = self.hdr_imgs["exposures"]["all"].copy()
                self.hdr_imgs["exposures"]["number"] = 0
                print("Picked exposures", self.hdr_imgs)
                self.aec_tab.aec_check.setChecked(False)
                cv2.imwrite(
                    f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}_base.{self.pic_tab.filetype.currentText()}",
                    new_cv_img
                )
            else:
                nearest_exposure = min(self.hdr_imgs["exposures"]["all"], key=lambda x: abs(x - new_exposure))
                if nearest_exposure == self.hdr_imgs["exposures"]["left"][0]:
                    self.hdr_imgs[new_exposure] = new_cv_img
                    self.hdr_imgs["exposures"]["number"] += 1
                    self.hdr_imgs["exposures"]["left"].pop(0)
                    print("Taken", self.hdr_imgs["exposures"]["number"], "images")
                else:
                    print("Waiting for exposure switch from", new_exposure, "to", self.hdr_imgs["exposures"]["left"][0])
            if self.hdr_imgs["exposures"]["number"] == len(self.hdr_imgs["exposures"]["all"]):
                print(f"All {len(self.hdr_imgs) - 1} HDR exposures captured, dispatching thread to process them")
                print("Captured exposures", list(self.hdr_imgs.keys())[1:])
                print("Desired exposures", self.hdr_imgs["exposures"]["all"])
                thread = threading.Thread(target=self.process_hdr, daemon=True)
                thread.start()
                self.aec_tab.aec_check.setChecked(True)
                self.mode_tabs.setEnabled(True)
                self.rec_button.setEnabled(True)
                self.pic_tab.hdr.setChecked(False)
                self.pic_tab.hdr.setEnabled(False)
                if self.pic_tab.preview_check.isChecked():
                    self.switch_config("preview")
                return
            else:
                self.picam2.controls.ExposureTime = self.hdr_imgs["exposures"]["left"][0]
                thread = threading.Thread(target=self.rec_button.clicked.emit, daemon=True)
                thread.start()


    def process_hdr(self):
        del self.hdr_imgs["exposures"]
        img_list = []
        exposures = []
        for k, v in self.hdr_imgs.items():
            img_list.append(v)
            exposures.append(int(k))
        exposures = np.array(exposures, dtype=np.float32)
        exposures /= 1e6
        print("Ready")
        tonemap = cv2.createTonemap(gamma=self.pic_tab.hdr_gamma.value())
        self.pic_tab.hdr_label.setText("HDR (Processing)")

        mean_image = np.average(np.array(img_list), axis=0)
        mean_8bit = mean_image.astype('uint8')
        cv2.imwrite(
            f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}_mean.{self.pic_tab.filetype.currentText()}",
            mean_8bit
        )
        del mean_image, mean_8bit
        print("Mean Done")

        merge_debevec = cv2.createMergeDebevec()
        hdr_debevec = merge_debevec.process(img_list, times=exposures.copy())
        res_debevec = tonemap.process(hdr_debevec.copy())
        res_debevec_8bit = np.clip(res_debevec * 255, 0, 255).astype('uint8')
        cv2.imwrite(
            f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}_debevec.{self.pic_tab.filetype.currentText()}",
            res_debevec_8bit
        )
        del merge_debevec, hdr_debevec, res_debevec, res_debevec_8bit
        print("Debevec Done")

        merge_robertson = cv2.createMergeRobertson()
        hdr_robertson = merge_robertson.process(img_list, times=exposures.copy())
        res_robertson = tonemap.process(hdr_robertson.copy())
        res_robertson_8bit = np.clip(res_robertson * 255, 0, 255).astype('uint8')
        cv2.imwrite(
            f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}_robertson.{self.pic_tab.filetype.currentText()}",
            res_robertson_8bit
        )
        del merge_robertson, hdr_robertson, res_robertson, res_robertson_8bit
        print("Robertson Done")

        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
        cv2.imwrite(
            f"{self.pic_tab.filename.text() if self.pic_tab.filename.text() else 'test'}_mertens.{self.pic_tab.filetype.currentText()}",
            res_mertens_8bit
        )
        del merge_mertens, res_mertens, res_mertens_8bit
        print("Mertens Done")

        print("Saved All HDR Images")
        self.hdr_imgs = {"exposures": None}
        self.pic_tab.hdr.setEnabled(True)
        self.pic_tab.hdr_label.setText("HDR")
    
    def toggle_hidden_controls(self):
        self.tabs.setHidden(not self.tabs.isHidden())
        new_width = self.window.width() + (-self.tabs.width() if self.tabs.isHidden() else self.tabs.width())
        self.window.resize(new_width, self.window.height())
        self.hide_button.setText("<" if self.tabs.isHidden() else ">")
        
       
    def run(self):
        app = QApplication([])
        self.window = QWidget()
        self.window.setWindowTitle("Qt Picamera2 App")
        bg_colour = self.window.palette().color(QPalette.Background).getRgb()[:3]
        self.qpicamera2 = QGlPicamera2(self.picam2, width=800, height=600, keep_ar=True, bg_colour=bg_colour)
        self.rec_button = QPushButton("Take Photo")
        self.rec_button.clicked.connect(self.on_rec_button_clicked)
        self.qpicamera2.done_signal.connect(self.capture_done)

        self.tabs = QTabWidget()
        self.img_tab = self.IMGTab(self)
        self.pan_tab = self.panTab(self)
        self.aec_tab = self.AECTab(self)
        self.info_tab = QLabel(alignment=Qt.AlignTop)
        self.other_tab = self.otherTab(self)
        self.hide_button = QPushButton(">")
        self.hide_button.clicked.connect(self.toggle_hidden_controls)
        self.hide_button.setMaximumSize(50, 400)

        self.mode_tabs = QTabWidget()
        self.pic_tab = self.picTab(self)
        self.vid_tab = self.vidTab(self)
        self.mode_tabs.currentChanged.connect(self.on_mode_change)

        self.tabs.setFixedWidth(400)
        self.mode_tabs.setFixedWidth(400)
        layout_h = QHBoxLayout()
        layout_v = QVBoxLayout()

        self.tabs.addTab(self.img_tab, "Image Tuning")
        self.tabs.addTab(self.pan_tab, "Pan/Zoom")
        self.tabs.addTab(self.aec_tab, "AEC/AWB")
        self.tabs.addTab(self.info_tab, "Info")
        self.tabs.addTab(self.other_tab, "Other")

        self.mode_tabs.addTab(self.pic_tab, "Still Capture")
        self.mode_tabs.addTab(self.vid_tab, "Video")

        layout_v.addWidget(self.mode_tabs)
        layout_v.addWidget(self.rec_button)

        layout_h.addLayout(layout_v)
        layout_h.addWidget(self.qpicamera2)
        layout_h.addWidget(self.hide_button)
        layout_h.addWidget(self.tabs)

        self.window.resize(1600, 600)
        self.window.setLayout(layout_h)

        self.window.closeEvent = self.close_event

        self.pic_tab.apply_settings()

        self.window.show()
        app.exec()
        

    def close_event(self, event):
        self.picam2.close()
        event.accept()

    class logControlSlider(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QHBoxLayout()
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(self.layout)

            self.slider = QSlider(Qt.Horizontal)
            self.box = QDoubleSpinBox()

            self.valueChanged = self.box.valueChanged
            self.valueChanged.connect(lambda: self.setValue(self.value()))
            self.slider.valueChanged.connect(self.updateValue)

            self.layout.addWidget(self.box)
            self.layout.addWidget(self.slider)

            self.precision = self.box.singleStep()
            self.slider.setSingleStep(1)
            self.minimum = 0.0
            self.maximum = 2.0

        @property
        def points(self):
            return int(1.0 / self.precision) * 2

        def boxToSlider(self, val=None):
            if val is None:
                val = self.box.value()
            if val == 0:
                return 0
            else:
                center = self.points // 2
                scaling = center / np.log2(self.maximum)
                return round(np.log2(val) * scaling) + center

        def sliderToBox(self, val=None):
            if val is None:
                val = self.slider.value()
            if val == 0:
                return 0
            else:
                center = self.points // 2
                scaling = center / np.log2(self.maximum)
                return round(2**((val - center) / scaling), int(-np.log10(self.precision)))

        def updateValue(self):
            self.blockAllSignals(True)
            if self.box.value() != self.sliderToBox():
                self.box.setValue(self.sliderToBox())
            self.blockAllSignals(False)
            self.valueChanged.emit(self.value())

        def redrawSlider(self):
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.points)
            self.slider.setValue(self.boxToSlider())

        def setSingleStep(self, val):
            self.box.setSingleStep(val)
            self.precision = val

        def setValue(self, val, emit=False):
            self.blockAllSignals(True)
            self.box.setValue(val)
            self.redrawSlider()
            self.blockAllSignals(False)
            if emit:
                self.valueChanged.emit(self.value())

        def setMinimum(self, val):
            self.box.setMinimum(val)
            self.minimum = val
            self.redrawSlider()

        def setMaximum(self, val):
            self.box.setMaximum(val)
            self.maximum = val
            self.redrawSlider()

        def blockAllSignals(self, y):
            self.box.blockSignals(y)
            self.slider.blockSignals(y)

        def value(self):
            return self.box.value()

    class controlSlider(QWidget):
        def __init__(self, camera_app, box_type=float):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QHBoxLayout()
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(self.layout)

            self.slider = QSlider(Qt.Horizontal)
            if box_type == float:
                self.box = QDoubleSpinBox()
            else:
                self.box = QSpinBox()

            self.valueChanged = self.box.valueChanged
            self.valueChanged.connect(lambda: self.setValue(self.value()))
            self.slider.valueChanged.connect(self.updateValue)

            self.layout.addWidget(self.box)
            self.layout.addWidget(self.slider)

            self.precision = self.box.singleStep()
            self.slider.setSingleStep(1)

        def updateValue(self):
            self.blockAllSignals(True)
            if self.box.value() != self.slider.value() * self.precision:
                self.box.setValue(self.slider.value() * self.precision)
            self.blockAllSignals(False)
            self.valueChanged.emit(self.value())

        def setSingleStep(self, val):
            self.box.setSingleStep(val)
            self.precision = val

        def setValue(self, val, emit=False):
            self.blockAllSignals(True)
            if val is None:
                val = 0
            self.box.setValue(val)
            self.slider.setValue(int(val / self.precision))
            self.blockAllSignals(False)
            if emit:
                self.valueChanged.emit(self.value())

        def setMinimum(self, val):
            self.box.setMinimum(val)
            self.slider.setMinimum(int(val / self.precision))

        def setMaximum(self, val):
            self.box.setMaximum(val)
            self.slider.setMaximum(int(val / self.precision))

        def blockAllSignals(self, y):
            self.box.blockSignals(y)
            self.slider.blockSignals(y)

        def value(self):
            return self.box.value()

    class panTab(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QFormLayout()
            self.setLayout(self.layout)

            self.label = QLabel((
                "Pan and Zoom Controls\n"
                "To zoom in/out, scroll up/down in the display below\n"
                "To pan, click and drag in the display below"),
                alignment=Qt.AlignCenter)
            self.zoom_text = QLabel("Current Zoom Level: 1.0", alignment=Qt.AlignCenter)
            self.pan_display = camera_app.panZoomDisplay(camera_app)
            self.pan_display.updated.connect(lambda: self.zoom_text.setText(
                f"Current Zoom Level: {self.pan_display.zoom_level:.1f}x"))

            self.layout.addRow(self.label)
            self.layout.addRow(self.zoom_text)
            self.layout.addRow(self.pan_display)
            self.layout.setAlignment(self.pan_display, Qt.AlignCenter)

    class panZoomDisplay(QWidget):
        updated = pyqtSignal()

        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.setMinimumSize(201, 151)
            _, full_img, _ = self.camera_app.picam2.camera_controls['ScalerCrop']
            self.scale = 200 / full_img[2]
            self.zoom_level_ = 1.0
            self.max_zoom = 7.0
            self.zoom_step = 0.1
            self.picam2 = camera_app.picam2

        @property
        def zoom_level(self):
            return self.zoom_level_

        @zoom_level.setter
        def zoom_level(self, val):
            if val != self.zoom_level:
                self.zoom_level_ = val
                self.setZoom()

        def setZoomLevel(self, val):
            self.zoom_level = val

        def paintEvent(self, event):
            painter = QPainter()
            painter.begin(self)
            _, full_img, _ = self.picam2.camera_controls['ScalerCrop']
            self.scale = 200 / full_img[2]
            scaled_full_img = [int(i * self.scale) for i in full_img]
            origin = scaled_full_img[:2]
            scaled_full_img[:2] = [0, 0]
            painter.drawRect(*scaled_full_img)
            scaled_scaler_crop = [int(i * self.scale) for i in self.camera_app.scaler_crop]
            scaled_scaler_crop[0] -= origin[0]
            scaled_scaler_crop[1] -= origin[1]
            painter.drawRect(*scaled_scaler_crop)
            painter.end()
            self.updated.emit()

        def draw_centered(self, pos):
            center = [int(i / self.scale) for i in pos]
            _, full_img, _ = self.picam2.camera_controls['ScalerCrop']
            w = self.camera_app.scaler_crop[2]
            h = self.camera_app.scaler_crop[3]
            x = center[0] - w // 2 + full_img[0]
            y = center[1] - h // 2 + full_img[1]
            new_scaler_crop = [x, y, w, h]

            new_scaler_crop[1] = max(new_scaler_crop[1], full_img[1])
            new_scaler_crop[1] = min(new_scaler_crop[1], full_img[1] + full_img[3] - new_scaler_crop[3])
            new_scaler_crop[0] = max(new_scaler_crop[0], full_img[0])
            new_scaler_crop[0] = min(new_scaler_crop[0], full_img[0] + full_img[2] - new_scaler_crop[2])
            self.camera_app.scaler_crop = tuple(new_scaler_crop)
            self.picam2.controls.ScalerCrop = self.camera_app.scaler_crop
            self.update()

        def mouseMoveEvent(self, event):
            pos = event.pos()
            pos = (pos.x(), pos.y())
            self.draw_centered(pos)

        def setZoom(self):
            if self.zoom_level < 1:
                self.zoom_level = 1.0
            if self.zoom_level > self.max_zoom:
                self.zoom_level = self.max_zoom
            factor = 1.0 / self.zoom_level
            _, full_img, _ = self.picam2.camera_controls['ScalerCrop']
            current_center = (self.camera_app.scaler_crop[0] + self.camera_app.scaler_crop[2] // 2, self.camera_app.scaler_crop[1] + self.camera_app.scaler_crop[3] // 2)
            w = int(factor * full_img[2])
            h = int(factor * full_img[3])
            x = current_center[0] - w // 2
            y = current_center[1] - h // 2
            new_scaler_crop = [x, y, w, h]
            new_scaler_crop[1] = max(new_scaler_crop[1], full_img[1])
            new_scaler_crop[1] = min(new_scaler_crop[1], full_img[1] + full_img[3] - new_scaler_crop[3])
            new_scaler_crop[0] = max(new_scaler_crop[0], full_img[0])
            new_scaler_crop[0] = min(new_scaler_crop[0], full_img[0] + full_img[2] - new_scaler_crop[2])
            self.camera_app.scaler_crop = tuple(new_scaler_crop)
            self.picam2.controls.ScalerCrop = self.camera_app.scaler_crop
            self.update()

        def wheelEvent(self, event):
            zoom_dir = np.sign(event.angleDelta().y())
            self.zoom_level += zoom_dir * self.zoom_step
            self.setZoom()

    class AECTab(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QFormLayout()
            self.setLayout(self.layout)

            self.aec_check = QCheckBox("AEC")
            self.aec_check.setChecked(True)
            self.aec_check.stateChanged.connect(self.aec_update)
            self.aec_meter = QComboBox()
            self.aec_meter.addItems(["Centre Weighted", "Spot", "Matrix"])
            self.aec_meter.currentIndexChanged.connect(self.aec_update)
            self.aec_constraint = QComboBox()
            self.aec_constraint.addItems(["Default", "Highlight"])
            self.aec_constraint.currentIndexChanged.connect(self.aec_update)
            self.aec_exposure = QComboBox()
            self.aec_exposure.addItems(["Normal", "Short", "Long"])
            self.aec_exposure.currentIndexChanged.connect(self.aec_update)
            self.exposure_val = self.camera_app.controlSlider(self.camera_app)
            self.exposure_val.valueChanged.connect(self.aec_update)
            self.exposure_val.setSingleStep(0.1)
            self.exposure_time = QSpinBox()
            self.exposure_time.setSingleStep(1000)
            self.analogue_gain = QDoubleSpinBox()
            self.analogue_label = QLabel()
            self.aec_apply = QPushButton("Apply Manual Values")
            self.aec_apply.setEnabled(False)
            self.aec_apply.clicked.connect(self.aec_manual_update)
            self.exposure_time.valueChanged.connect(lambda: self.aec_apply.setEnabled(self.exposure_time.isEnabled()))
            self.analogue_gain.valueChanged.connect(lambda: self.aec_apply.setEnabled(self.exposure_time.isEnabled()))

            self.awb_check = QCheckBox("AWB")
            self.awb_check.setChecked(True)
            self.awb_check.stateChanged.connect(self.awb_update)
            self.awb_mode = QComboBox()
            self.awb_mode.addItems([
                "Auto", "Incandescent", "Tungsten", "Fluorescent",
                "Indoor", "Daylight", "Cloudy"
            ])
            self.awb_mode.currentIndexChanged.connect(self.awb_update)
            self.colour_gain_r = QDoubleSpinBox()
            self.colour_gain_r.setSingleStep(0.1)
            self.colour_gain_r.valueChanged.connect(self.awb_update)
            self.colour_gain_b = QDoubleSpinBox()
            self.colour_gain_b.setSingleStep(0.1)
            self.colour_gain_b.valueChanged.connect(self.awb_update)

            self.reset()
            self.aec_update()
            self.awb_update()
            self.aec_apply.setEnabled(False)

            self.layout.addRow(self.aec_check)
            self.layout.addRow("AEC Metering Mode", self.aec_meter)
            self.layout.addRow("AEC Constraint Mode", self.aec_constraint)
            self.layout.addRow("AEC Exposure Mode", self.aec_exposure)
            self.layout.addRow("Exposure Value", self.exposure_val)
            self.layout.addRow("Exposure Time/\u03bcs", self.exposure_time)
            self.layout.addRow("Gain", self.analogue_gain)
            self.layout.addRow(self.analogue_label)
            self.layout.addRow(self.aec_apply)

            self.layout.addRow(self.awb_check)
            self.layout.addRow("AWB Mode", self.awb_mode)
            self.layout.addRow("Red Gain", self.colour_gain_r)
            self.layout.addRow("Blue Gain", self.colour_gain_b)

        def reset(self):
            self.aec_check.setChecked(True)
            self.awb_check.setChecked(True)
            self.exposure_time.setValue(10000)
            self.analogue_gain.setValue(1.0)
            self.colour_gain_r.setValue(1.0)
            self.colour_gain_b.setValue(1.0)

        @property
        def aec_dict(self):
            ret = {
                "AeEnable": self.aec_check.isChecked(),
                "AeMeteringMode": self.aec_meter.currentIndex(),
                "AeConstraintMode": self.aec_constraint.currentIndex(),
                "AeExposureMode": self.aec_exposure.currentIndex(),
                "ExposureValue": self.exposure_val.value(),
                "ExposureTime": self.exposure_time.value(),
                "AnalogueGain": self.analogue_gain.value()
            }
            if self.aec_check.isChecked():
                del ret["ExposureTime"]
                del ret["AnalogueGain"]
            return ret

        def aec_update(self):
            self.exposure_val.setMinimum(self.camera_app.picam2.camera_controls["ExposureValue"][0])
            self.exposure_val.setMaximum(self.camera_app.picam2.camera_controls["ExposureValue"][1])
            self.exposure_time.setMinimum(self.camera_app.picam2.camera_controls["ExposureTime"][0])
            self.exposure_time.setMaximum(self.camera_app.picam2.camera_controls["ExposureTime"][1])
            self.analogue_gain.setMinimum(self.camera_app.picam2.camera_controls["AnalogueGain"][0])
            self.analogue_label.setText(f"Analogue up to {self.camera_app.picam2.camera_controls['AnalogueGain'][1]:.2f}, then digital beyond")

            self.aec_meter.setEnabled(self.aec_check.isChecked())
            self.aec_constraint.setEnabled(self.aec_check.isChecked())
            self.aec_exposure.setEnabled(self.aec_check.isChecked())
            self.exposure_val.setEnabled(self.aec_check.isChecked())
            self.exposure_time.setEnabled(not self.aec_check.isChecked())
            self.analogue_gain.setEnabled(not self.aec_check.isChecked())
            if self.aec_check.isChecked():
                self.aec_apply.setEnabled(False)
            self.camera_app.picam2.set_controls(self.aec_dict)

        def aec_manual_update(self):
            if not self.aec_check.isChecked():
                self.aec_update()
            self.aec_apply.setEnabled(False)

        @property
        def awb_dict(self):
            ret = {
                "AwbEnable": self.awb_check.isChecked(),
                "AwbMode": self.awb_mode.currentIndex(),
                "ColourGains": [self.colour_gain_r.value(), self.colour_gain_b.value()]
            }
            if self.awb_check.isChecked():
                del ret["ColourGains"]
            return ret

        def awb_update(self):
            self.colour_gain_r.setMinimum(self.camera_app.picam2.camera_controls["ColourGains"][0] + 0.01)
            self.colour_gain_r.setMaximum(self.camera_app.picam2.camera_controls["ColourGains"][1])
            self.colour_gain_b.setMinimum(self.camera_app.picam2.camera_controls["ColourGains"][0] + 0.01)
            self.colour_gain_b.setMaximum(self.camera_app.picam2.camera_controls["ColourGains"][1])

            self.colour_gain_r.setEnabled(not self.awb_check.isChecked())
            self.colour_gain_b.setEnabled(not self.awb_check.isChecked())
            self.camera_app.picam2.set_controls(self.awb_dict)

    class IMGTab(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QFormLayout()
            self.setLayout(self.layout)

            self.ccm = QDoubleSpinBox()
            self.ccm.valueChanged.connect(self.img_update)
            self.saturation = self.camera_app.logControlSlider(self.camera_app)
            self.saturation.valueChanged.connect(self.img_update)
            self.saturation.setSingleStep(0.1)
            self.contrast = self.camera_app.logControlSlider(self.camera_app)
            self.contrast.valueChanged.connect(self.img_update)
            self.contrast.setSingleStep(0.1)
            self.sharpness = self.camera_app.logControlSlider(self.camera_app)
            self.sharpness.valueChanged.connect(self.img_update)
            self.sharpness.setSingleStep(0.1)
            self.brightness = self.camera_app.controlSlider(self.camera_app)
            self.brightness.setSingleStep(0.1)
            self.brightness.valueChanged.connect(self.img_update)
            self.noise_reduction = QComboBox()
            self.noise_reduction.addItems(["Off", "Fast", "High Quality", "Minimal", "ZSL"])
            self.noise_reduction.currentIndexChanged.connect(self.img_update)
            self.reset_button = QPushButton("Reset")
            self.reset_button.clicked.connect(self.reset)

            self.reset()
            self.img_update()

            self.layout.addRow("Saturation", self.saturation)
            self.layout.addRow("Contrast", self.contrast)
            self.layout.addRow("Sharpness", self.sharpness)
            self.layout.addRow("Brightness", self.brightness)
            self.layout.addRow(self.reset_button)

        @property
        def img_dict(self):
            return {
                "Saturation": self.saturation.value(),
                "Contrast": self.contrast.value(),
                "Sharpness": self.sharpness.value(),
                "Brightness": self.brightness.value(),
            }

        def reset(self):
            self.saturation.setValue(self.camera_app.picam2.camera_controls["Saturation"][2], emit=True)
            self.contrast.setValue(self.camera_app.picam2.camera_controls["Contrast"][2], emit=True)
            self.sharpness.setValue(self.camera_app.picam2.camera_controls["Sharpness"][2], emit=True)
            self.brightness.setValue(self.camera_app.picam2.camera_controls["Brightness"][2], emit=True)

        def img_update(self):
            self.saturation.setMinimum(self.camera_app.picam2.camera_controls["Saturation"][0])
            self.saturation.setMaximum(6.0)
            self.contrast.setMinimum(self.camera_app.picam2.camera_controls["Contrast"][0])
            self.contrast.setMaximum(6.0)
            self.sharpness.setMinimum(self.camera_app.picam2.camera_controls["Sharpness"][0])
            self.sharpness.setMaximum(self.camera_app.picam2.camera_controls["Sharpness"][1])
            self.brightness.setMinimum(self.camera_app.picam2.camera_controls["Brightness"][0])
            self.brightness.setMaximum(self.camera_app.picam2.camera_controls["Brightness"][1])
            self.camera_app.picam2.set_controls(self.img_dict)

    class otherTab(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QFormLayout()
            self.setLayout(self.layout)

            all_controls = self.camera_app.picam2.camera_controls.keys()
            other_controls = []
            for control in all_controls:
                if control not in self.camera_app.implemented_controls and control not in self.camera_app.ignore_controls:
                    other_controls.append(control)
            self.fields = {}
            for control in other_controls:
                widget = self.camera_app.controlSlider(self.camera_app, box_type=type(self.camera_app.picam2.camera_controls[control][0]))
                print(control)
                print(type(self.camera_app.picam2.camera_controls[control][0]))
                print(self.camera_app.picam2.camera_controls[control][0])
                widget.setMinimum(self.camera_app.picam2.camera_controls[control][0])
                widget.setMaximum(self.camera_app.picam2.camera_controls[control][1])
                widget.setValue(self.camera_app.picam2.camera_controls[control][2])
                widget.valueChanged.connect(self.other_update)
                self.fields[control] = widget

            for k, v in self.fields.items():
                self.layout.addRow(k, v)

            print("Other controls", other_controls)

        @property
        def other_dict(self):
            ret = {}
            for k, v in self.fields.items():
                ret[k] = v.value()
            return ret

        def other_update(self):
            self.camera_app.picam2.set_controls(self.other_dict)

    class vidTab(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QFormLayout()
            self.setLayout(self.layout)

            self.filename = QLineEdit()
            self.filetype = QComboBox()
            self.filetype.addItems(["mp4", "mkv", "ts", "mov", "avi", "h264"])
            self.quality_box = QComboBox()
            self.quality_box.addItems(["Very Low", "Low", "Medium", "High", "Very High"])
            self.framerate = QSpinBox()
            self.framerate.valueChanged.connect(self.vid_update)
            self.framerate.setMinimum(1)
            self.framerate.setMaximum(500)
            self.actual_framerate = QLabel()
            self.resolution_w = QSpinBox()
            self.resolution_w.setMaximum(self.camera_app.picam2.sensor_resolution[0])
            self.resolution_h = QSpinBox()
            self.resolution_h.setMaximum(min(self.camera_app.picam2.sensor_resolution[1], 1080))
            self.raw_format = QComboBox()
            self.raw_format.addItem("Default")
            self.raw_format.addItems([f'{x["format"].format} {x["size"]}, {x["fps"]:.0f}fps' for x in self.camera_app.picam2.sensor_modes])
            self.apply_button = QPushButton("Apply")
            self.apply_button.clicked.connect(self.apply_settings)

            resolution = QWidget()
            res_layout = QHBoxLayout()
            res_layout.addWidget(self.resolution_w)
            res_layout.addWidget(QLabel("x"), alignment=Qt.AlignHCenter)
            res_layout.addWidget(self.resolution_h)
            resolution.setLayout(res_layout)

            self.layout.addRow("Name", self.filename)
            self.layout.addRow("File Type", self.filetype)
            self.layout.addRow("Quality", self.quality_box)
            self.layout.addRow("Frame Rate", self.framerate)
            self.layout.addRow(self.actual_framerate)
            self.layout.addRow("Resolution", resolution)
            self.layout.addRow("Sensor Mode", self.raw_format)
            self.layout.addRow(self.apply_button)

            self.reset()

        @property
        def quality(self):
            qualities = {
                "Very Low": Quality.VERY_LOW,
                "Low": Quality.LOW,
                "Medium": Quality.MEDIUM,
                "High": Quality.HIGH,
                "Very High": Quality.VERY_HIGH
            }
            return qualities[self.quality_box.currentText()]

        @property
        def sensor_mode(self):
            configs = [None]
            for mode in self.camera_app.picam2.sensor_modes:
                configs.append({"size": mode["size"], "format": mode["format"].format})
            return configs[self.raw_format.currentIndex()]

        @property
        def frametime(self):
            return self.frametime_

        @frametime.setter
        def frametime(self, value):
            self.frametime_ = value
            self.actual_framerate.setText(f"Actual Framerate: {1e6 / self.frametime:.1f}fps")

        @property
        def vid_dict(self):
            return {
                "FrameRate": self.framerate.value()
            }

        def vid_update(self):
            if self.isVisible():
                self.camera_app.picam2.set_controls(self.vid_dict)
            else:
                print("Not setting vid controls when not visible")

        def reset(self):
            self.quality_box.setCurrentIndex(2)
            self.framerate.setValue(30)
            self.resolution_h.setValue(720)
            self.resolution_w.setValue(1280)
            self.camera_app.picam2.video_configuration = self.camera_app.picam2.create_video_configuration(
                main={"size": (self.resolution_w.value(), self.resolution_h.value())},
                raw=self.sensor_mode
            )

        def apply_settings(self):
            self.camera_app.picam2.video_configuration = self.camera_app.picam2.create_video_configuration(
                main={"size": (self.resolution_w.value(), self.resolution_h.value())},
                raw=self.sensor_mode
            )
            self.camera_app.switch_config("video")

    class picTab(QWidget):
        def __init__(self, camera_app):
            super().__init__()
            self.camera_app = camera_app
            self.layout = QFormLayout()
            self.setLayout(self.layout)

            self.filename = QLineEdit()
            self.filetype = QComboBox()
            self.filetype.addItems(["jpg", "png", "bmp", "gif", "raw"])
            self.resolution_w = QSpinBox()
            self.resolution_w.setMaximum(self.camera_app.picam2.sensor_resolution[0])
            self.resolution_w.valueChanged.connect(lambda: self.apply_button.setEnabled(True))
            self.resolution_h = QSpinBox()
            self.resolution_h.setMaximum(self.camera_app.picam2.sensor_resolution[1])
            self.resolution_h.valueChanged.connect(lambda: self.apply_button.setEnabled(True))
            self.raw_format = QComboBox()
            self.raw_format.addItem("Default")
            self.raw_format.addItems([f'{x["format"].format} {x["size"]}' for x in self.camera_app.picam2.sensor_modes])
            self.raw_format.currentIndexChanged.connect(self.update_options)
            self.preview_format = QComboBox()
            self.preview_format.currentIndexChanged.connect(lambda: self.apply_button.setEnabled(True))
            self.preview_check = QCheckBox()
            self.preview_check.setChecked(True)
            self.preview_check.stateChanged.connect(self.apply_settings)
            self.preview_warning = QLabel("WARNING: Preview and Capture modes have different fields of view")
            self.preview_warning.setWordWrap(True)
            self.preview_warning.hide()
            self.hdr_label = QLabel("HDR")
            self.hdr = QCheckBox()
            self.hdr.setChecked(False)
            self.hdr.setEnabled(cv_present)
            if cv_present:
                self.hdr.stateChanged.connect(self.pic_update)
                self.num_hdr = QSpinBox()
                self.num_hdr.setRange(3, 8)
                self.stops_hdr_above = QSpinBox()
                self.stops_hdr_above.setRange(1, 10)
                self.stops_hdr_below = QSpinBox()
                self.stops_hdr_below.setRange(1, 10)
                self.hdr_gamma = QDoubleSpinBox()
                self.hdr_gamma.setSingleStep(0.1)
            self.apply_button = QPushButton("Apply")
            self.apply_button.clicked.connect(self.apply_settings)
            self.apply_button.setEnabled(False)

            resolution = QWidget()
            res_layout = QHBoxLayout()
            res_layout.addWidget(self.resolution_w)
            res_layout.addWidget(QLabel("x"), alignment=Qt.AlignHCenter)
            res_layout.addWidget(self.resolution_h)
            resolution.setLayout(res_layout)

            self.pic_update()
            self.update_options()
            self.reset()

            self.layout.addRow("Name", self.filename)
            self.layout.addRow("File Type", self.filetype)
            self.layout.addRow("Resolution", resolution)
            self.layout.addRow("Sensor Mode", self.raw_format)
            self.layout.addRow("Enable Preview Mode", self.preview_check)
            self.layout.addRow(self.preview_warning)
            self.layout.addRow("Preview Mode", self.preview_format)
            if cv_present:
                self.layout.addRow(self.hdr_label, self.hdr)
                self.layout.addRow("Number of HDR frames", self.num_hdr)
                self.layout.addRow("Number of HDR stops above", self.stops_hdr_above)
                self.layout.addRow("Number of HDR stops below", self.stops_hdr_below)
                self.layout.addRow("HDR Gamma Setting", self.hdr_gamma)
            else:
                self.layout.addRow(QLabel("HDR unavailable - install opencv to try it out"))

            self.layout.addRow(self.apply_button)

        @property
        def sensor_mode(self):
            configs = [{}]
            for mode in self.camera_app.picam2.sensor_modes:
                configs.append({"size": mode["size"], "format": mode["format"].format})
            return configs[self.raw_format.currentIndex()]

        @property
        def preview_mode(self):
            configs = [self.sensor_mode]
            for mode in self.preview_modes:
                configs.append({"size": mode["size"], "format": mode["format"].format})
            return configs[self.preview_format.currentIndex()]

        @property
        def pic_dict(self):
            return {
                "FrameDurationLimits": self.camera_app.picam2.camera_controls["FrameDurationLimits"][0:2]
            }

        def pic_update(self):
            if cv_present:
                self.stops_hdr_above.setEnabled(self.hdr.isChecked())
                self.stops_hdr_below.setEnabled(self.hdr.isChecked())
                self.num_hdr.setEnabled(self.hdr.isChecked())
                self.hdr_gamma.setEnabled(self.hdr.isChecked())
            if self.isVisible():
                self.camera_app.picam2.set_controls(self.pic_dict)
            else:
                print("Not setting pic controls when not visible")

        def reset(self):
            self.resolution_h.setValue(self.camera_app.picam2.still_configuration.main.size[1])
            self.resolution_w.setValue(self.camera_app.picam2.still_configuration.main.size[0])
            if cv_present:
                self.hdr_gamma.setValue(2.2)
            self.camera_app.picam2.still_configuration = self.camera_app.picam2.create_still_configuration(
                main={"size": (self.resolution_w.value(), self.resolution_h.value())},
                **self.camera_app.still_kwargs,
                raw=self.sensor_mode,
            )

        def update_options(self):
            self.apply_button.setEnabled(True)
            try:
                self.resolution_w.setValue(self.sensor_mode["size"][0])
                self.resolution_h.setValue(self.sensor_mode["size"][1])
            except KeyError:
                self.resolution_h.setValue(self.camera_app.picam2.still_configuration.main.size[1])
                self.resolution_w.setValue(self.camera_app.picam2.still_configuration.main.size[0])

            preview_index = self.preview_format.currentIndex()
            if preview_index < 0:
                preview_index = 0
            if self.sensor_mode:
                crop_limits = self.camera_app.picam2.sensor_modes[self.raw_format.currentIndex() - 1]["crop_limits"]
            else:
                crop_limits = (0, 0, *self.camera_app.picam2.sensor_resolution)
            self.preview_format.clear()
            self.preview_format.addItem("Same as capture")
            self.preview_modes = []
            for mode in self.camera_app.picam2.sensor_modes:
                if mode["crop_limits"] == crop_limits:
                    self.preview_format.addItem(f'{mode["format"].format} {mode["size"]}')
                    self.preview_modes.append(mode)
            try:
                self.preview_format.setCurrentIndex(preview_index)
            except IndexError:
                self.preview_format.setCurrentIndex(0)

        def apply_settings(self):
            self.camera_app.hide_button.setEnabled(self.preview_check.isChecked())

            self.camera_app.picam2.still_configuration = self.camera_app.picam2.create_still_configuration(
                main={"size": (self.resolution_w.value(), self.resolution_h.value())},
                **self.camera_app.still_kwargs,
                raw=self.sensor_mode,
            )
            self.camera_app.picam2.preview_configuration = self.camera_app.picam2.create_preview_configuration(
                main={"size": (
                    self.camera_app.qpicamera2.width(), int(self.camera_app.qpicamera2.width() * (self.resolution_h.value() / self.resolution_w.value()))
                )},
                raw=self.preview_mode
            )
            self.preview_format.setEnabled(self.preview_check.isChecked())

            if self.preview_check.isChecked():
                self.camera_app.switch_config("still")
                _, current_crop, _ = self.camera_app.picam2.camera_controls['ScalerCrop']
                self.camera_app.switch_config("preview")
                _, preview_crop, _ = self.camera_app.picam2.camera_controls['ScalerCrop']
                if current_crop != preview_crop:
                    print("Preview and Still configs have different aspect ratios")
                    self.preview_warning.show()
                else:
                    self.preview_warning.hide()
            else:
                self.camera_app.switch_config("still")
                self.preview_warning.hide()
            self.apply_button.setEnabled(False)


if __name__ == "__main__":
    

    camera_app = RPiCameraApp()
    camera_app.run()

