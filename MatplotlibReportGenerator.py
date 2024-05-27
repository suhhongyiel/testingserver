import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


class MatplotlibReportGenerator():
    def __init__(self, plot_num=4, figsize=(8, 10)):
        self.plot_num = plot_num
        self.plots = [] # ax object list
        self.fig = plt.figure(figsize=figsize)  # 적당한 크기의 figure 생성
        self.gs = gridspec.GridSpec(plot_num, 1)
        # gridspec은 아직 정확한 사용법은 모르지만, subplot을 그리는데 도움을 주는 것 같음.
        for ax in range(plot_num):
            self.plots.append(self.fig.add_subplot(self.gs[ax]))

    def get_ax(self, target):
        """
        return ax object to draw plot.
        Each ax represents one section or plot in the report(pdf).
        example:
        report = ReportGenerator()
        report.get_fig(0).plot(x, y)
        """
        return self.plots[target]

    def get_fig(self):
        """
        return fig object for more detailed customization
        """
        return self.fig

    def savepdf(self, filename="report"):
        if filename.find(".") != -1:
            print("Warning: filename doesn't need extension. It always saves as pdf.")
            filename = filename.split(".")[0]

        with PdfPages(filename + ".pdf") as pdf:
            self.fig.tight_layout()
            pdf.savefig(self.fig)
            plt.close()
        return None

    def set_title(self, title, fontsize=16, horizontalalignment="center"):
        self.fig.suptitle(title, fontsize=fontsize, horizontalalignment=horizontalalignment)
