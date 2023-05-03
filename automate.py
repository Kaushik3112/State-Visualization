from automan.api import Problem, Automator, Simulation
from automan.api import mdict, opts2path, filter_cases
import numpy as np
import matplotlib.pyplot as plt
from visualization import state_converter, animate_bloch
# import moviepy.video.io.ImageSequenceClip

opts = mdict(number=[4, 6, 8, 10],
             state_req=[False, True])

class state(Problem):
    def get_name(self):
        return 'state'
    
    def setup(self):
        base_cmd = 'python main.py -o $output_dir//data.npz'
        self.cases = [
            Simulation(
            root=self.input_path(opts2path(kw)),
            base_command=base_cmd,
            **kw
            )
            for kw in opts
        ]

    def run(self):
        self.make_output_dir()
        method1 = [False]
        method2 = [True]
        for method in method1:
            filtered_cases1 = filter_cases(self.cases, state_req=method)
            i = 1
            for case in filtered_cases1:
                stdout = self.input_path(case.name, 'data.npz')
                data = np.load(stdout)
                files = data.files
                w_range = data[files[0]]
                X_vals = data[files[1]]
                plt.figure(i)
                plt.plot(w_range, X_vals, label=f"Excitation for n={case.params['number']}")
                plt.title(f"Excitation for n={case.params['number']}")
                plt.xlabel("Frequency offset in Hz")
                plt.ylabel("Excitation x direction")
                plt.savefig(self.output_path(f"{case.params['number']}exc.png"))
                i = i+1

        for method in method2:
            filtered_cases2 = filter_cases(self.cases, state_req=method)

            for case in filtered_cases2:
                stdout = self.input_path(case.name, 'data.npz')
                data = np.load(stdout)
                files = data.files
                states = data[files[-1]]
                state_list = state_converter(states)
                animate_bloch(state_list, save_all=True, path=self.output_path(f"tmp{case.params['number']}"))
                # image_folder = 'tmp'
                # video_name = 
                # fps = 10
                # images = [os.path.join(image_folder, 'anim_%02d.png' % i)
                #         for i in range(100)]
                # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images,
                #                                                             fps=fps)
                # clip.write_videofile(video_name, logger=None)



if __name__ == "__main__":
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[state]
    )

    automator.run()