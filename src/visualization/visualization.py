import numpy as np
import shutil
import time
import torch
import os
import json
import requests
import traceback
import warnings

from shutil import copyfile
from visdom import Visdom

class VisdomVisualizer:

    def __init__(self, env_name='main', port=1074, log_file : str = None, json_files_root : str = None):
        """Initialize visdom visualizer

        Args:
            env_name (str, optional): the environment name of the server. Defaults to 'main'.
            port (int, optional): the port number. Defaults to 1074.
            log_file (str, optional): the log filename to recover visdom file
            json_files_root (str, optional): json file root to save windows dicts

        Raises:
            TimeoutError: [description]
        """
        self.viz = VisdomCustom(server="localhost", port=port, log_to_filename=log_file, \
                          raise_exceptions=False)

        print("Setup visualization : available at http://localhost:{:d}".format(port))
        self.env = env_name
        self.log_file = log_file
        self.ckpt_root = json_files_root
        self.ckpt_windows_format = os.path.join(json_files_root, "windows_{:s}.json")
        self.ckpt_events_format = os.path.join(json_files_root, "events_{:s}.log")
        self.windows = {}


    def plot(self, X : torch.Tensor, Y : torch.Tensor, xlabel : str, ylabel : str, legend : str, title : str):
        """Plots lines given X and Y on visdom server and plotly style

        Args:
            X (torch.Tensor): The X values
            Y (torch.Tensor): The Y values
            xlabel (str): the label for x axis
            ylabel (str): the label for y axis
            legend (str): the legend of the curve
            title (str): the title of the window
        """
        if title not in self.windows:
            self.windows[title] = self.viz.line(X=X,
                                                 Y=Y,
                                                 env=self.env,
                                                 opts=dict(
                                                     legend=[legend],
                                                     title=title,
                                                     xlabel=xlabel,
                                                     ylabel=ylabel
                                                 ))
        else:
            self.viz.line(X=X,
                          Y=Y,
                          env=self.env,
                          win=self.windows[title],
                          name=legend,
                          update='append')


    def save_vis(self, epoch : str):
        """Save windows in a json file

        Args:
            epochs (str): epoch number
        """

        # Save env
        self.viz.save([self.env])

        # Copy log file
        log_file_name = self.ckpt_events_format.format(epoch)
        copyfile(self.log_file, log_file_name)

        # Save windows
        json_file_name = self.ckpt_windows_format.format(epoch)
        with open(json_file_name,"w") as f:
            json.dump(self.windows, f)


    def load_vis(self, epoch : str):
        """Load states of windows at a given epoch

        Args:
            epoch (str): the epoch to load
        """

        # Get ckpt log file
        log_file_name = self.ckpt_events_format.format(epoch)

        if os.path.exists(log_file_name):
            copyfile(log_file_name, self.log_file)

        # Get windows
        json_file_name = self.ckpt_windows_format.format(epoch)
        if os.path.exists(json_file_name):
            with open(json_file_name,"r") as f:
                self.windows = json.load(f)

        # Replay log
        self.viz.replay_log(self.log_file)

    def reset_env(self):
        """Resets Visdom logs, delete ckpt visdom log directory
        """
        if os.path.exists(self.log_file):
            # Reinitialize visdom log file
            with open(self.log_file, "r+") as f:
                f.truncate(0)
        else:
            # Create the file
            open(self.log_file, 'a').close()

        # Clean visdom log directory
        shutil.rmtree(self.ckpt_root)
        os.mkdir(self.ckpt_root)

        # Delete env
        self.viz.close(win=None, env=self.env)

    def show_images(self, title_name, images, nrow=4):
        """This methods shows a grid of images

        Args:
            title_name (str): the title of the grid, it is also the key name of the window
            images (Tensor): 3d/4d Tensor representing a grid of images
            nrow (int, optional): The number of columns in the grid. Defaults to 4.
        """
        if images.shape[0] > 0:
            if title_name not in self.windows:
                self.windows[title_name] = self.viz.images(images, nrow, env=self.env, opts=dict(caption=title_name,
                                                                                                log=False))
            else:
                self.viz.images(images, nrow, env=self.env, win=self.windows[title_name], opts=dict(caption=title_name,
                                                                                                log=False))

    def show_image(self, image, title_name):
        """This method shows a single image

        Args:
            image (Tensor): 2d/3d Tensor image
            title_name (str): The title of the plot, it is also the key name of the window
        """
        if title_name not in self.windows:
            self.windows[title_name] = self.viz.image(image, env=self.env, opts=dict(caption=title_name,
                                                                                     store_history=True,
                                                                                     log=False))
        else:
            self.viz.image(image, env=self.env, win=self.windows[title_name], opts=dict(caption=title_name,
                                                                                        store_history=True,
                                                                                        log=False))

    def show_figure(self, title_name, class_figure):
        """Given a figure in figures.py. Plots it in visdom server

        Args:
            title_name (str): The title of the window
            class_figure (FigurePlot): The figurePlot to show
        """
        image = class_figure.to_torch()
        self.show_image(image, title_name)

"""
We redefine the Visdom class to redefine the _send method save of Visdom.
Since we save log from visdom server, we observed a lot of space was consumed by logging images.
But we don't need them, so we add an option allowing the user to not log some entries.
"""
def get_rand_id():
    return str(hex(int(time.time() * 10000000))[2:])

class VisdomCustom(Visdom):
    """We redefine the Visdom class for visualiation purpose.
    The code is widely inspired from:
    https://github.com/fossasia/visdom
    """

    def __init__(
        self,
        server='https://google-github.herokuapp.com:443/http/localhost',
        endpoint='events',
        port=8097,
        base_url='/',
        ipv6=True,
        http_proxy_host=None,
        http_proxy_port=None,
        env='main',
        send=True,
        raise_exceptions=None,
        use_incoming_socket=True,
        log_to_filename=None,
        username=None,
        password=None,
        proxies=None,
        offline=False,
        use_polling=False,
    ):
        super().__init__(server=server, endpoint=endpoint,port=port,base_url=base_url,ipv6=ipv6,
            http_proxy_host=http_proxy_host,http_proxy_port=http_proxy_port,env=env,send=send,
            raise_exceptions=raise_exceptions,use_incoming_socket=use_incoming_socket,
            log_to_filename=log_to_filename,username=username,password=password,proxies=proxies,
            offline=offline,use_polling=use_polling)

    def _send(self, msg, endpoint='events', quiet=False, from_log=False, create=True):
        """
        This function sends specified JSON request to the Tornado server. This
        function should generally not be called by the user, unless you want to
        build the required JSON yourself. `endpoint` specifies the destination
        Tornado server endpoint for the request.
        If `create=True`, then if `win=None` in the message a new window will be
        created with a random name. If `create=False`, `win=None` indicates the
        operation should be applied to all windows.
        """
        if msg.get('eid', None) is None:
            msg['eid'] = self.env

        # TODO investigate send use cases, then deprecate
        if not self.send:
            return msg, endpoint

        if 'win' in msg and msg['win'] is None and create:
            msg['win'] = 'window_' + get_rand_id()

        # We allow opts to have a log option, if false we don't log
        if not from_log and msg.get('opts', {}).get('log', True):
            self._log(msg, endpoint)

        # We remove log key from opts to avoid erros
        msg.get('opts', {}).pop('log', None)

        if self.offline:
            # If offline, don't even try to post
            return msg['win'] if 'win' in msg else True

        try:
            return self._handle_post(
                "{0}:{1}{2}/{3}".format(self.server, self.port,
                                        self.base_url, endpoint),
                data=json.dumps(msg),
            )
        except (
            requests.RequestException, requests.ConnectionError,
            requests.Timeout
        ):
            if self.raise_exceptions:
                raise ConnectionError("Error connecting to Visdom server")
            else:
                if self.raise_exceptions is None:
                    warnings.warn(
                        "Visdom is eventually changing to default to raising "
                        "exceptions rather than ignoring/printing. This change"
                        " is expected to happen by July 2018. Please set "
                        "`raise_exceptions` to False to retain current "
                        "behavior.",
                        PendingDeprecationWarning
                    )
                if not quiet:
                    print("Exception in user code:")
                    print('-' * 60)
                    traceback.print_exc()
                return False
