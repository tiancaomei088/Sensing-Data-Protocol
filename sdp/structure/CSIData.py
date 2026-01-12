class CSIData:
    def __init__(self, file_name: str):
        """
        Initializes the CSIData object with the provided data.
        """
        self.file_name = file_name
        self.frames = []

    def add_frame(self, frame):
        """
        A CSI frame to the CSIData object.
        Usually one frame contains info of one timestamp of received signal.
        """
        self.frames.append(frame)
