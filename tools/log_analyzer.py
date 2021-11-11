import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from common import time_utils
import matplotlib.dates as md
import datetime as dt

log_dir_path = "./logs/tmp/sim/"
vehicle_log_file = "vehicle.log"
customer_log_file = "customer.log"
score_log_file = "score.log"
summary_log_file = "summary.log"

vehicle_log_cols = [
    "t",
    "id",
    "lat",
    "lon",
    "speed",
    "status",
    "destination_lat",
    "destination_lon",
    "type",
    "travel_dist",
    "price_per_travel_m",
    "price_per_wait_min",
    "gas_price",
    "assigned_customer_id",
    "time_to_destination",
    "idle_duration",
    "current_capacity",
    "max_capacity",
    "driver_base_per_trip",
    "mileage",
    "agent_type",
]


customer_log_cols = ["t", "id", "status", "waiting_time"]

summary_log_cols = [
    "readable_time",        ####Human readable time
    "t",                    ####Unix time
    "n_vehicles_OnDuty",    ####On duty vehicles
    "n_vehicles_Occupied",  ####Fully occupied vehicles
    "n_requests",           ####Total number of requests
    "n_requests_assigned",  ####Number of requests assigned
    "n_rejected_requests",  ####Number of Rejected Requests
    "n_accepted_commands",  ####Number of Requests Accepted by customers
    "average_wt",           ####Average Wait for all customers
    "avg_earnings",         ####Average Earnings per vehicle
    "avg_cost",             ####Average Cost per vehicle
    "avg_profit_dqn",       ####Average Profit per vehicle using dqn agent
    "avg_profit_dummy",     ####Average Profit per vehicle using dummy agent
    "avg_total_dist",       ####Average total distance travelled by vehicles (dqn or dummy?)
    "avg_cap",              ####Average SEATS occupancy per vehicle
    "avg_idle_time",        ####Average idle time per vehicle
]

score_log_cols = [
    "t",
    "vehicle_id",
    "working_time",
    "earning",
    "idle",
    "cruising",
    "occupied",
    "assigned",
    "offduty",
]


class LogAnalyzer(object):
    """This class helps analyze experimental results.
    The simulator should have output '.log' files in a given logging directory.
    The following log files may be parsed by this LogAnalyzer:
    summary.log, vehicle.log, customer.log, score.log
    """

    def __init__(self):
        pass

    def load_log(self, path, cols, max_num, skip_minutes=0):
        """Parses .log files into pd.DataFrame
        Args:
            path:           (str) Directory where sim logs are present.
            cols:           (list) List of column names for a given log file.
            max_num:        (int) Max number of logs to parse (if multiple experiments were run).
            skip_minutes:   (int) Number of minutes to skip in logs (from the top).
        Returns:
            df:             (pd.DataFrame) Logs are now returned as a DataFrame for easy manipulation.
        """
        df = pd.read_csv(path, names=cols)
        dfs = [df]
        for i in range(1, max_num):
            # pdb.set_trace()
            path_ = path + "." + str(i)
            try:
                df = pd.read_csv(path_, names=cols)
                dfs.append(df)
            except IOError:
                break
        df = pd.concat(dfs)
        df = df[df.t >= df.t.min() + skip_minutes * 60]
        return df

    def load_vehicle_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used to obtain vehicle logs as a DataFrame"""
        return self.load_log(
            log_dir_path + vehicle_log_file, vehicle_log_cols, max_num, skip_minutes
        )

    def load_customer_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used to obtain customer logs as a DataFrame"""
        return self.load_log(
            log_dir_path + customer_log_file, customer_log_cols, max_num, skip_minutes
        )

    def load_summary_log(self, log_dir_path, max_num=100, skip_minutes=0):
        print("Summary: ", log_dir_path + summary_log_file)
        """Used to obtain summary logs as a DataFrame"""
        return self.load_log(
            log_dir_path + summary_log_file, summary_log_cols, max_num, skip_minutes
        )

    def _load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used as a helper function for load_score_log"""
        return self.load_log(
            log_dir_path + score_log_file, score_log_cols, max_num, skip_minutes
        )

    def load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used to obtain score logs as a DataFrame"""
        df = self._load_score_log(log_dir_path, max_num, skip_minutes)
        # total_seconds = (df.t.max() - df.t.min() + 3600 * 24)
        # n_days = total_seconds / 3600 / 24
        # df = df[df.t == df.t.max()]
        # df["working_hour"] = (total_seconds - df.offduty) / n_days / 3600
        df["working_hour"] = (df.working_time - df.offduty) / 3600
        df["cruising_hour"] = (df.cruising + df.assigned) / 3600
        df["occupancy_rate"] = df.occupied / (df.working_hour * 3600) * 100
        # df["reward"] = (
        #     df.earning
        #     - (df.cruising + df.assigned + df.occupied) * DRIVING_COST / settings.TIMESTEP
        #     - (df.working_time - df.offduty) * WORKING_COST / settings.TIMESTEP
        # )
        df["revenue_per_hour"] = df.earning / df.working_hour

        return df

    def get_customer_status(self, customer_df, bin_width=300):
        """ Customer Status (discretized by time0 """
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = (
            customer_df.groupby(["time_bin", "status"])
            .size()
            .reset_index()
            .pivot(index="time_bin", columns="status", values=0)
            .fillna(0)
        )
        df = df.rename(columns={2: "ride_on", 4: "rejected"})
        df["total"] = sum([x for _, x in df.iteritems()])
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def get_customer_waiting_time(self, customer_df, bin_width=300):
        """ Customer Waiting time (discretized time) """
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = customer_df[customer_df.status == 2].groupby("time_bin").waiting_time.mean()
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def add_time_bin(self, df, bin_width):
        """Helper function to discretize time from minutes into bins of 'bin_width' """
        start_time = df.t.min()
        return ((df.t - start_time) / bin_width).astype(int) * int(bin_width) + start_time

    def plot_summary(self, paths, labels, plt):
        """Plotting of experiment summaries
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot.
        """
        plt.figure(figsize=(12, 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for i, path in enumerate(paths):
            summary = self.load_summary_log(path)
            # print(summary)
            summary["t"] = (summary.t / 3600).astype(int) * 3600
            summary = summary.groupby("t").mean().reset_index()
            # summary.t = [time_utils.get_local_datetime(t) for t in summary.t]
            # print(summary.t)
            summary.t = [dt.datetime.fromtimestamp(t) for t in summary.t]
            print(summary.t)

            plt.subplot(len(paths), 2, +i * 2 + 1)
            # print(summary.t)
            plt.plot(summary.t, summary.n_requests, label="request")

            plt.plot(
                summary.t, summary.n_rejected_requests, label="Reject", linestyle=":"
            )

            plt.plot(summary.t, summary.n_accepted_commands, label="Accepted", alpha=0.7)

            plt.ylabel("count/minute")
            plt.title(labels[0])
            plt.xlabel("simulation time (yy-mm-dd hr:min:sec)")
            plt.xticks(rotation=25)
            plt.ylim([0, 610])
            ax = plt.gca()
            xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            ax.set_xticks(summary.t)
            ax.xaxis.set_major_formatter(xfmt)

            if i != len(paths) - 1:
                plt.xticks([])
            if i == 0:
                plt.legend(loc="upper right")

            # plt.savefig("Accepts_Rejects.png")

            plt.subplot(len(paths), 2, i * 2 + 2)
            plt.title(labels[1])
            plt.plot(summary.t, summary.n_vehicles_OnDuty, label="OnDuty")
            plt.plot(
                summary.t, summary.n_vehicles_Occupied, label="Occupied", linestyle=":"
            )
            plt.ylabel("# of vehicles")
            # plt.xlabel("simulation time (mm-dd hh)")
            plt.xlabel("simulation time (yy-mm-dd hh:min:sec)")
            plt.xticks(rotation=25)
            plt.ylim([0, 10100])
            ax = plt.gca()
            xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            ax.set_xticks(summary.t)
            ax.xaxis.set_major_formatter(xfmt)
            if i != len(paths) - 1:
                plt.xticks([])
            if i == 0:
                plt.legend(loc="upper right")
                # plt.subplot(313)
            # plt.plot(summary.t, summary.average_wt, alpha=1.0)
            # plt.ylim([0, 450])
            # plt.ylabel("waiting time (s)")
            plt.xticks(rotation=25)
            # plt.savefig("Occupied_OnDuty.png")
        return plt

    def plot_metrics_ts(self, paths, labels, plt):
        """Plotting of experiment Scores
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot
        """
        plt.figure(figsize=(12, 4))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            score["t"] = ((score.t - score.t.min()) / 3600).astype(int)
            plt.subplot(131)
            plt.ylabel("revenue ($/h)")
            plt.scatter(score.t, score.revenue_per_hour, alpha=0.5, label=label)
            plt.ylim([0, 50])
            plt.subplot(132)
            plt.ylabel("cruising time (h/day)")
            plt.scatter(score.t, score.cruising_hour, alpha=0.5, label=label)

        plt.legend()
        return plt

    def plot_metrics(self, paths, labels, plt):
        """Plotting of misc. experiment metrics (uses vehicle, customer, and score logs.)
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot.
        """
        data = []
        plt.figure(figsize=(12, 3))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            c = self.load_customer_log(p, skip_minutes=60)

            plt.subplot(141)
            plt.xlabel("revenue ($/h)")
            plt.hist(
                score.revenue_per_hour, bins=100, range=(18, 42), alpha=0.5, label=labels[0]
            )
            plt.yticks([])
            plt.title(labels[0])

            plt.subplot(142)
            plt.xlabel("working time (h/day)")
            plt.hist(score.working_hour, bins=100, range=(17, 23), alpha=0.5, label=labels[1])
            plt.yticks([])
            plt.title(labels[1])

            plt.subplot(143)
            plt.xlabel("cruising time (h/day)")
            plt.hist(
                score.cruising_hour, bins=100, range=(1.9, 7.1), alpha=0.5, label=labels[2]
            )
            plt.yticks([])
            plt.title(labels[2])

            # plt.subplot(234)
            # plt.xlabel("occupancy rate")
            # plt.hist(score.occupancy_rate, bins=100, range=(55, 75), alpha=0.5, label=label)
            # plt.yticks([])
            #
            # plt.subplot(235)
            # plt.xlabel("total reward / day")
            # plt.hist(score.reward, bins=100, range=(-10, 410), alpha=0.5, label=label)
            # plt.yticks([])

            plt.subplot(144)
            plt.xlabel("waiting time (s)")
            plt.hist(
                c[c.status == 2].waiting_time,
                bins=500,
                range=(0, 650),
                alpha=0.5,
                label=labels[3],
            )
            plt.yticks([])
            plt.title(labels[3])

            x = {}
            x["00_reject_rate"] = float(len(c[c.status == 4])) / len(c) * 100
            x["01_revenue/hour"] = score.revenue_per_hour.mean()
            x["02_occupancy_rate"] = score.occupancy_rate.mean()
            x["03_cruising/day"] = score.cruising_hour.mean()
            x["04_working/day"] = score.working_hour.mean()
            x["05_waiting_time"] = c[c.status == 2].waiting_time.mean()

            x["11_revenue/hour(std)"] = score.revenue_per_hour.std()
            x["12_occupancy_rate(std)"] = score.occupancy_rate.std()
            x["13_cruising/day(std)"] = score.cruising_hour.std()
            x["14_working/day(std)"] = score.working_hour.std()
            x["15_waiting_time(std)"] = c[c.status == 2].waiting_time.std()
            data.append(x)

        plt.legend()

        df = pd.DataFrame(data, index=labels)
        return plt, df
        """Plotting of misc. experiment metrics (uses vehicle, customer, and score logs.)
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot.
        """
        data = []
        plt.figure(figsize=(12, 3))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            c = self.load_customer_log(p, skip_minutes=60)

            plt.subplot(141)
            plt.xlabel("revenue ($/h)")
            plt.hist(
                score.revenue_per_hour, bins=100, range=(18, 42), alpha=0.5, label=label
            )
            plt.yticks([])

            plt.subplot(142)
            plt.xlabel("working time (h/day)")
            plt.hist(score.working_hour, bins=100, range=(17, 23), alpha=0.5, label=label)
            plt.yticks([])

            plt.subplot(143)
            plt.xlabel("cruising time (h/day)")
            plt.hist(
                score.cruising_hour, bins=100, range=(1.9, 7.1), alpha=0.5, label=label
            )
            plt.yticks([])

            # plt.subplot(234)
            # plt.xlabel("occupancy rate")
            # plt.hist(score.occupancy_rate, bins=100, range=(55, 75), alpha=0.5, label=label)
            # plt.yticks([])
            #
            # plt.subplot(235)
            # plt.xlabel("total reward / day")
            # plt.hist(score.reward, bins=100, range=(-10, 410), alpha=0.5, label=label)
            # plt.yticks([])

            plt.subplot(144)
            plt.xlabel("waiting time (s)")
            plt.hist(
                c[c.status == 2].waiting_time,
                bins=500,
                range=(0, 650),
                alpha=0.5,
                label=label,
            )
            plt.yticks([])

            x = {}
            x["00_reject_rate"] = float(len(c[c.status == 4])) / len(c) * 100
            x["01_revenue/hour"] = score.revenue_per_hour.mean()
            x["02_occupancy_rate"] = score.occupancy_rate.mean()
            x["03_cruising/day"] = score.cruising_hour.mean()
            x["04_working/day"] = score.working_hour.mean()
            x["05_waiting_time"] = c[c.status == 2].waiting_time.mean()

            x["11_revenue/hour(std)"] = score.revenue_per_hour.std()
            x["12_occupancy_rate(std)"] = score.occupancy_rate.std()
            x["13_cruising/day(std)"] = score.cruising_hour.std()
            x["14_working/day(std)"] = score.working_hour.std()
            x["15_waiting_time(std)"] = c[c.status == 2].waiting_time.std()
            data.append(x)

        plt.legend()
        df = pd.DataFrame(data, index=labels)
        return plt, df
