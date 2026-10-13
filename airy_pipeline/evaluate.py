import os
import cv2
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

from IPython.display import display
from airy_pipeline.methods.evaluate import to_smooth, to_msds
from airy_pipeline.methods import where, to_directory


class Evaluate:
    def __init__(self, file=None, beads=None, psf=None, directory=None, config=None, start=0,
                 end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
        """
        Runs the evaluate user interface for quality
        control of pipeline outputs.

        Parameters
        ----------
        file : str, optional
            Path to CZI file.
        beads : str, optional
            Path to TXT file of bead files.
        psf : str, optional
            Path to H5 file of PSFs.
        directory : str, optional
            Path for pipeline outputs.
        config : str, optional
            Path to YAML file.
        start : int, optional
            Starting frame (python indexed).
        end : int, optional
            Ending frame (python indexed).
        nproc : int, optional
            Number of processes used.
        gpus : int, optional
            Number of GPUs to used.
        mode : str, optional
            Module specific mode.
        other : any, optional
            For pipeline testing.
        force: bool, optional
            If forcing overwrite.
        """
        # argument handling
        if directory is None:
            if file is not None:
                self.directory = to_directory(file)
            else:
                print("Directory is required for this module!") 
                return
        else:
            self.directory = directory

        # hold current index, track data, and video paths
        self.idx = 0
        self.tracks = []
        self.videos = []
        self.frame = 0

        # read saved annotations
        self.write_path = os.path.join(self.directory, "corrections", "annotations.csv")
        exists = os.path.exists(self.write_path)
        if exists:
            # read CSV to lists
            info = np.loadtxt(self.write_path, delimiter=",", skiprows=1, dtype=int)
            self.nuclei = info[:,0].tolist()
            self.subtracks = info[:,1].tolist()
            self.start = info[:,2].tolist()
            self.end = (info[:,3]-1).tolist()
            self.decision = info[:,4].tolist()
        else:
            # make empty for later
            self.nuclei = []
            self.subtracks = []
            self.start = []
            self.end = []
            self.decision = []

        # iterate through nuclei looking for tracks
        self.distances = np.empty(0)  # hold 3D distances
        nuclei_path = os.path.join(self.directory, "nuclei")
        for item in os.listdir(nuclei_path):
            if item.isnumeric():
                # get path to tracks        
                tracks_path = os.path.join(nuclei_path, item, "tracks")
                for subitem in os.listdir(tracks_path):
                    if subitem[:-4].isnumeric():
                        # read track and make paths
                        track = np.loadtxt(os.path.join(tracks_path, subitem), delimiter=",", skiprows=1)
                        video_path = os.path.join(self.directory, "movies", "evaluate", item, f"{subitem[:-4]}.mp4")

                        # add if sixty or more valid frames
                        if np.sum(~np.isnan(track[:,23])) >= 60:
                            # add track data, video path, and distances
                            self.tracks.append(track)                        
                            self.videos.append(video_path)
                            self.distances = np.concatenate((self.distances, track[:,23]))

                            # set initials for sliders/subsets
                            if not exists:
                                self.nuclei.append(int(item))
                                self.subtracks.append(int(subitem[:-4]))
                                self.start.append(0)
                                self.end.append(len(track)-1)
                                self.decision.append(-1)   

        # ---------------- colors ----------------
        self.colors = {"red": "#FD4441",
                       "orange": "#F28642",
                       "yellow": "#F2D024",
                       "green": "#31E86E",
                       "blue": "#30ABCA",
                       "navy": "#2D3588",
                       "purple": "#8944B8",
                       "gray": "#8A8A8A",
                       "black": "#000000"}
        
        # ---------------- plots ----------------
        plt.ioff()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={"height_ratios": [1, 2]}, dpi=75)

        # plot 3D distance histogram (movie)
        self.hist_cache = {}
        median = np.nanmedian(self.distances)
        self.axes[0,0].hist(self.distances, bins=100, linewidth=2, color=self.colors["navy"], histtype="step")
        self.axes[0,0].set_yticks([])
        self.axes[0,0].set_xlabel("Distances (nm)")
        self.axes[0,0].set_xlim([-50, 1550])
        self.hist_cache[0] = self.axes[0,0].axvline(median, color=self.colors["navy"], linewidth=2,
                                                    label=f"Median (movie): {median:.2f} (nm)")
        # plot 3D distance histogram (track)
        self.hist_cache[1] = self.axes[0,0].axvline(median, color=self.colors["gray"], linewidth=2,
                                                    label=f"Median (track): {median:.2f} (nm)")
        self.hist_cache[2] = self.axes[0,0].twinx()
        self.hist_cache[2].set_yticks([])
        self.axes[0,0].legend()

        # plot MSDs (for track)
        self.msds_cache = {}
        self.msds_cache[2], = self.axes[0,1].loglog([], [], linewidth=2, label="X", color=self.colors["yellow"])
        self.msds_cache[1], = self.axes[0,1].loglog([], [], linewidth=2, label="Y", color=self.colors["navy"])
        self.msds_cache[0], = self.axes[0,1].loglog([], [], linewidth=2, label="Z", color=self.colors["green"])
        self.axes[0,1].set_xlabel("Lags (frames)")
        self.axes[0,1].set_ylabel("MSD (nm²)")
        self.axes[0,1].legend(fontsize=12, ncol=3, columnspacing=0.6)

        # plot distances (for track)
        self.distances_cache = {}
        self.distances_cache[0], = self.axes[1,0].plot([], [], color=self.colors["gray"])
        self.distances_cache[1], = self.axes[1,0].plot([], [], color=self.colors["orange"])
        self.axes[1,0].set_xlim([-20, 740])
        self.axes[1,0].set_ylim([-50, 1050])
        self.axes[1,0].set_xlabel("Time (frames)")
        self.axes[1,0].set_ylabel("Distance (nm)")
        
        # plot intensities (for track)
        self.intensities_cache = {}
        self.axes[1,1].patch.set_visible(False)
        self.axes[1,1].axhline(0, color=self.colors["black"], linewidth=2)
        self.intensities_cache[0], = self.axes[1,1].plot([], [], color=self.colors["blue"])
        self.intensities_cache[1], = self.axes[1,1].plot([], [], color=self.colors["orange"])
        self.intensities_cache[2] = self.axes[1,1].twinx()
        self.intensities_cache[3], = self.intensities_cache[2].plot([], [], color=self.colors["navy"])
        self.intensities_cache[4], = self.intensities_cache[2].plot([], [], color=self.colors["green"])
        self.axes[1,1].set_zorder(self.intensities_cache[2].get_zorder() + 1)
        self.axes[1,1].set_xlim([-20, 740])
        self.axes[1,1].set_xlabel("Time (frames)")
        self.axes[1,1].set_ylabel("Intensity (au)")
        self.axes[1,1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))        

        # cache plot items
        self.replicated_cache = {}
        self.frame_cache, self.span_cache = {}, {}
        self.start_cache, self.end_cache = {}, {}
        for i in range(2):
            # do for distance and intensity plots
            self.replicated_cache[i] = self.axes[1,i].axvline(0, color=self.colors["yellow"], linewidth=2)
            self.frame_cache[i] = self.axes[1,i].axvline(0, color=self.colors["red"], linewidth=2)
            self.start_cache[i] = self.axes[1,i].axvline(0, color=self.colors["black"], linestyle="--", linewidth=2)
            self.end_cache[i] = self.axes[1,i].axvline(0, color=self.colors["black"], linestyle="--", linewidth=2)
            
            # handle twin axis
            ax = self.axes[1,i] if i == 0 else self.intensities_cache[2]
            self.span_cache[i] = ax.axvspan(0, 0, color=self.colors["yellow"], alpha=0.3, zorder=0)
            
        # configure plotting
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.resizable = False
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        
        # ---------------- widgets ----------------
        # video
        self.video = widgets.Image(format="jpg", width=1080)
        
        # buttons
        layout = widgets.Layout(width="100px", height="30px")
        self.keeping_btn = widgets.Button(description="Keeping ✓", layout=layout)
        self.removing_btn = widgets.Button(description="Removing ✗", layout=layout)
        self.previous_btn = widgets.Button(description="← Previous", layout=layout)
        self.next_btn = widgets.Button(description="Next →", layout=layout)

        # sliders
        layout = widgets.Layout(width="420px")
        self.frame_slr = widgets.IntSlider(min=0, max=0, readout=False, layout=layout,
                                           continuous_update=False)
        self.start_end_slr = widgets.IntRangeSlider(min=0, max=0, readout=False,
                                                    layout=layout, continuous_update=False)
        # entries
        layout = widgets.Layout(width="40px", height="30px")
        self.frame_txt = widgets.Text(continuous_update=False, layout=layout)
        self.start_txt = widgets.Text(continuous_update=False, layout=layout)
        self.end_txt = widgets.Text(continuous_update=False, layout=layout)
        self.track_txt = widgets.Text(continuous_update=False, layout=layout)

        # labels
        self.frame_lbl = widgets.Label(value="frame:")
        self.start_lbl = widgets.Label(value="start:")
        self.end_lbl = widgets.Label(value="end:")
        self.track_lbl = widgets.Label(value="track:")
        self.of_lbl= widgets.Label(value=f"of {len(self.tracks)}")
        
        # configure button actions
        self.keeping_btn.on_click(self.action_keeping_btn)
        self.removing_btn.on_click(self.action_removing_btn)
        self.previous_btn.on_click(self.action_previous_btn)
        self.next_btn.on_click(self.action_next_btn)

        # configure slider actions
        self.frame_slr.observe(self.action_frame_slr, names="value")
        self.start_end_slr.observe(self.action_start_end_slr, names="value")

        # configure entry actions
        self.track_txt.observe(self.action_track_txt, names="value")
        self.start_txt.observe(self.action_start_txt, names="value")
        self.end_txt.observe(self.action_end_txt, names="value")
        self.frame_txt.observe(self.action_frame_txt, names="value")

        # ---------------- display ----------------
        self.update_track()  # show first track
        
        # group inputs together
        space = widgets.Box(layout=widgets.Layout(height="15px", width="5px"))
        layout = widgets.Layout(justify_content="flex-end", width="100%")
        a = widgets.HBox([self.frame_lbl, self.frame_txt], layout=layout)
        b = widgets.HBox([self.start_lbl, self.start_txt, space, self.end_lbl, self.end_txt])
        c = widgets.HBox([self.track_lbl, self.track_txt, self.of_lbl], layout=layout)
        d = widgets.VBox([self.frame_slr, self.start_end_slr])
        e = widgets.HBox([self.previous_btn, self.next_btn, space, self.keeping_btn, self.removing_btn])
        inputs = widgets.HBox([widgets.VBox([a, b, c]), space, widgets.VBox([d, e])])
        
        # display elements
        layout = widgets.Layout(display="flex", flex_flow="column", align_items="center", width="100%")
        dsp = widgets.VBox([self.fig.canvas, self.video, inputs], layout=layout)
        display(dsp)

    def write(self):
        """ 
        Saves the interface state for all tracks.
        """
        # set CSV format
        fmt = 5*["%d"]
        header = "nucleus,track,start,end,decision"

        # save as CSV
        info = np.stack((self.nuclei, self.subtracks, self.start, self.end, self.decision), axis=1)
        info[:,3] += 1
        np.savetxt(self.write_path, info, delimiter=",", fmt=fmt, header=header, comments="")

    def action_keeping_btn(self, btn):
        """ 
        Defines how the "keeping" button widget
        acts when pressed.
        """
        # update colors and decision
        self.removing_btn.style.button_color = None
        if (self.decision[self.idx] == 0) or (self.decision[self.idx] == -1):
            self.keeping_btn.style.button_color = self.colors["green"]
            self.decision[self.idx] = 1            
        else:
            self.keeping_btn.style.button_color = None
            self.decision[self.idx] = -1

    def action_removing_btn(self, btn):
        """
        Defines how the "removing" button widget
        acts when pressed.
        """
        # update colors and decision
        self.keeping_btn.style.button_color = None
        if (self.decision[self.idx] == 1) or (self.decision[self.idx] == -1):
            self.removing_btn.style.button_color = self.colors["red"]
            self.decision[self.idx] = 0
        else:
            self.removing_btn.style.button_color = None
            self.decision[self.idx] = -1

    def action_previous_btn(self, btn):
        """
        Defines how the "previous" button widget
        acts when pressed.
        """
        # decrease the index and load
        if self.idx > 0:
            self.idx -= 1
            self.track_txt.value = str(self.idx + 1)
            self.update_track()
        
    def action_next_btn(self, btn):
        """
        Defines how the "next" button widget
        acts when pressed.
        """
        # increase the index and load
        if self.idx < (len(self.tracks) - 1):
            self.idx += 1
            self.track_txt.value = str(self.idx + 1)
            self.update_track()

    def action_track_txt(self, change):
        """
        Defines how the "track" text widget
        handles input.
        """
        try:
            # make sure in bounds
            idx = int(change["new"]) - 1
            if idx == self.idx:
                return
            elif (idx < 0) or (idx > len(self.tracks)-1):
                raise ValueError
            else:
                self.idx = idx
                self.update_track()

        except ValueError:
            self.track_txt.value = str(self.idx + 1)

    def action_frame_txt(self, change):
        """
        Defines how the "frame" text widget
        handles input.
        """
        try:
            # make sure in bounds
            frame = int(change["new"])
            if frame == self.frame:
                return
            elif (frame < 0) or (frame > len(self.track)):
                raise ValueError
            else:
                self.frame = frame
                self.update_frame()

        except ValueError:
            self.frame_txt.value = str(self.frame)
        finally:
            self.frame_slr.value = self.frame
            
    def action_frame_slr(self, change):
        """
        Defines how the "frame" slider
        handles input.
        """
        # get frame and update
        frame = change["new"]
        if frame != self.frame:
            self.frame = frame
            self.update_frame()
    
    def action_start_txt(self, change):
        """
        Defines how the "start" text widget
        handles input.
        """
        try:
            # make sure in bounds
            start = int(change["new"])
            if start == self.start[self.idx]:
                return
            elif (start < 0) or (start > len(self.track)) or (start > self.end[self.idx]):
                raise ValueError
            else:
                self.start[self.idx] = start
                self.update_start()

        except ValueError:
            self.start_txt.value = str(self.start[self.idx])
        finally:
            self.start_end_slr.value = (self.start[self.idx], self.end[self.idx])

    def action_end_txt(self, change):
        """
        Defines how the "end" text widget
        handles input.
        """
        try:
            # make sure in bounds
            end = int(change["new"])
            if end == self.end[self.idx]:
                return
            elif (end < 0) or (end > len(self.track)) or (end < self.start[self.idx]):
                raise ValueError
            else:
                self.end[self.idx] = end
                self.update_end()

        except ValueError:
            self.end_txt.value = str(self.end[self.idx])
        finally:
            self.start_end_slr.value = (self.start[self.idx], self.end[self.idx])
    
    def action_start_end_slr(self, change):
        """
        Defines how the "start" and "end" slider
        handles input.
        """
        # get start/end and update
        start, end = change["new"]
        if self.start[self.idx] != start:
            self.start[self.idx] = start
            self.update_start()
        if self.end[self.idx] != end:
            self.end[self.idx] = end
            self.update_end()
       
    def update_track(self):
        """
        Updates the interface to show the next track.
        """
        # pull track
        self.track = self.tracks[self.idx]

        # update slider ranges
        n = len(self.track)
        self.start_end_slr.value = (self.start[self.idx], self.end[self.idx])
        self.frame_slr.min = 0
        self.frame_slr.max = n-1
        self.start_end_slr.min = 0
        self.start_end_slr.max = n-1
        self.start_end_slr.value = (self.start[self.idx], self.end[self.idx])

        # read frames
        self.images = []
        cap = cv2.VideoCapture(self.videos[self.idx])
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode(".jpg", image)
            self.images.append(buffer.tobytes())
        cap.release()

        # update video
        frame = int((n-1)/2)
        self.frame = frame
        self.video.value = self.images[frame]

        # update distances/intensities
        t = np.arange(len(self.track))
        self.distances_cache[0].set_data(t, self.track[:,23])
        self.distances_cache[1].set_data(t, to_smooth(self.track[:,23], 3))
        self.intensities_cache[3].set_data(t, self.track[:,24])
        self.intensities_cache[4].set_data(t, to_smooth(self.track[:,24], 3))
        self.intensities_cache[0].set_data(t, self.track[:,25])
        self.intensities_cache[1].set_data(t, to_smooth(self.track[:,25], 3))

        # find where replicated (and update)
        replicated = where(self.track[:,22])
        if replicated is None:
            replicated = n
        for i in range(2):
            self.replicated_cache[i].set_xdata([replicated, replicated])

        # update distance histogram (track)
        self.hist_cache[2].clear()
        self.hist_cache[2].hist(self.track[:,23], bins=25, linewidth=2, color=self.colors["gray"], histtype="step")
        self.hist_cache[2].set_yticks([])

        # update track distance median
        median = np.nanmedian(self.track[:,23])
        self.hist_cache[1].set_xdata([median, median])
        self.hist_cache[1].set_label(f"Median (track): {median:.2f} (nm)")
        self.axes[0,0].legend()

        # update MSDs plot
        msds, _ = to_msds(self.track[:,18:21] - self.track[:,6:9])
        place = int(len(self.track)/2)
        for axis in range(3):
            self.msds_cache[axis].set_data(np.arange(1,place), msds[1:place,axis])
        self.axes[0,1].relim()
        self.axes[0,1].autoscale_view()

        # update intensities plot
        mean = np.nanmean(self.track[:,25])
        self.intensities_cache[0].set_label(f"Mean: {mean:.3f} (au)")
        self.axes[1,1].relim()
        self.axes[1,1].autoscale_view()
        self.axes[1,1].legend()
        self.intensities_cache[2].relim()
        self.intensities_cache[2].autoscale_view()

        # update interface and re-draw
        self.track_txt.value = str(self.idx + 1)
        self.update_frame(draw=False, write=False, show=False)
        self.update_start(draw=False, write=False, show=False)
        self.update_end(draw=False, write=False, show=False)
        self.update_decision(write=False)
        self.fig.canvas.draw_idle()
        self.write()
        self.start_end_slr.value = (self.start[self.idx], self.end[self.idx])

    def update_frame(self, draw=True, write=True, show=True):
        """
        Updates the video frame and frame
        markers on plots.
        """
        frame = self.frame
        self.frame_txt.value = str(frame)
        for i in range(2):
            self.frame_cache[i].set_xdata([frame, frame])
        if draw:
            self.fig.canvas.draw_idle()
        if write:
            self.write()
        if show:
            self.video.value = self.images[frame]

    def update_start(self, draw=True, write=True, show=True):
        """
        Updates the start markers on plots.
        """
        start, end = self.start[self.idx], self.end[self.idx]
        self.start_txt.value = str(start)
        for i in range(2):
            self.start_cache[i].set_xdata([start, start])
            self.span_cache[i].set_x(start)
            self.span_cache[i].set_width(end-start)
        if draw:
            self.fig.canvas.draw_idle()
        if write:
            self.write()
        if show:
            self.video.value = self.images[start]

    def update_end(self, draw=True, write=True, show=True):
        """
        Updates the end markers on plots.
        """
        start, end = self.start[self.idx], self.end[self.idx]
        self.end_txt.value = str(end)
        for i in range(2):
            self.end_cache[i].set_xdata([end, end])
            self.span_cache[i].set_width(end-start)
        if draw:
            self.fig.canvas.draw_idle()
        if write:
            self.write()
        if show:
            self.video.value = self.images[end]

    def update_decision(self, write=True):
        """
        Updates the keeping/removing buttons.
        """
        if self.decision[self.idx] == -1:
            self.keeping_btn.style.button_color = None
            self.removing_btn.style.button_color = None
        elif self.decision[self.idx] == 1:
            self.keeping_btn.style.button_color = self.colors["green"]
            self.removing_btn.style.button_color = None
        else:
            self.keeping_btn.style.button_color = None
            self.removing_btn.style.button_color = self.colors["red"]
        if write:
            self.write()
