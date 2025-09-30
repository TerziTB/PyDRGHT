import numpy as np
import pandas as pd

class DChar:
    """
    Class to characterize drought events in a time series.
    """

    def __init__(self, time_series, onset_threshold, recovery_threshold, min_drought_duration):
        """
        Initialize the DChar object.

        Parameters
        ----------
        time_series : pd.Series
            Time series data indexed by datetime or ordinal.
        onset_threshold : float
            Value below which a drought is considered to start.
        recovery_threshold : float
            Value above which a drought is considered to have recovered.
        min_drought_duration : int
            Minimum number of consecutive periods to consider a drought event.
        """
        self.ts = time_series.dropna()
        self.onset_threshold = onset_threshold
        self.recovery_threshold = recovery_threshold
        self.min_drought_duration = min_drought_duration
        self.features = None

    def calculate(self):
        """
        Calculate drought characteristics for the stored time series.

        Returns
        -------
        pandas.DataFrame
            DataFrame where each row corresponds to a drought event, with columns:
            - 'Duration' : int
                Length of the drought event (number of time steps).
            - 'Severity' : float
                Sum of the drought index values during the event.
            - 'Intensity' : float
                Mean of the drought index values during the event.
            - 'Date_Ini_Ev' : datetime-like
                Start date of the drought event.
            - 'Date_Fin_Ev' : datetime-like
                End date of the drought event.
            - 'Interarrival' : float or NaN
                Time (in steps) since the previous drought ended. NaN for the first/last event depending on padding.
            - 'Frequency_of_Occ' : float
                Percentage of the time series occupied by this drought event.
            - 'Recovery_Duration' : float or 'No recovery'
                Duration until recovery threshold is reached, or 'No recovery' if not recovered.
        """
        
        ts = self.ts
        duration, severity, intensity = [], [], []
        date_ini_ev, date_fin_ev, interarrival = [], [], []
        frequency, recovery_duration = [], []

        drought_event = False
        event_start, event_end, drought_duration = None, None, 0
        event_intensity, next_event_start = [], None
        total_length = len(ts)

        for i, value in enumerate(ts):
            if value <= self.onset_threshold:
                if not drought_event:
                    event_start = ts.index[i]
                    drought_event = True
                drought_duration += 1
                event_intensity.append(value)
            else:
                if drought_event:
                    if drought_duration >= self.min_drought_duration:
                        event_end = ts.index[i]
                        duration.append(drought_duration)
                        severity.append(sum(event_intensity))
                        date_ini_ev.append(event_start)
                        date_fin_ev.append(event_end)
                        intensity.append(np.mean(event_intensity))
                        frequency.append((drought_duration / total_length) * 100)

                        # Recovery calculation
                        recovery_start_index = ts.index.get_loc(event_end) + 1
                        recovery_end_index = None
                        for j in range(recovery_start_index, len(ts)):
                            if ts.iloc[j] <= self.onset_threshold:
                                next_event_start = ts.index[j]
                                break
                        for k in range(recovery_start_index, len(ts)):
                            if ts.iloc[k] >= self.recovery_threshold:
                                recovery_end_index = k
                                break
                        if recovery_end_index is not None and (
                            next_event_start is None or ts.index[recovery_end_index] < next_event_start
                        ):
                            recovery_duration.append(ts.index[recovery_end_index] - event_end + 1)
                        else:
                            recovery_duration.append("No recovery")

                    # Reset event tracking
                    drought_event = False
                    event_intensity = []
                    drought_duration = 0

        # Handle ongoing drought at end of series
        if drought_event and drought_duration >= self.min_drought_duration:
            event_end = ts.index[-1]
            duration.append(drought_duration)
            severity.append(sum(event_intensity))
            date_ini_ev.append(event_start)
            date_fin_ev.append(event_end)
            intensity.append(np.mean(event_intensity))
            frequency.append((drought_duration / total_length) * 100)

            recovery_start_index = ts.index.get_loc(event_end) + 1
            recovery_end_index = None
            for j in range(recovery_start_index, len(ts)):
                if ts.iloc[j] <= self.onset_threshold:
                    next_event_start = ts.index[j]
                    break
            for k in range(recovery_start_index, len(ts)):
                if ts.iloc[k] >= self.recovery_threshold:
                    recovery_end_index = k
                    break
            if recovery_end_index is not None and (
                next_event_start is None or ts.index[recovery_end_index] < next_event_start
            ):
                recovery_duration.append(ts.index[recovery_end_index] - event_end + 1)
            else:
                recovery_duration.append("No recovery")

        # Calculate interarrival times
        for j in range(1, len(date_ini_ev)):
            interarrival.append(date_ini_ev[j] - date_fin_ev[j - 1])
        if len(interarrival) < len(duration):
            interarrival = interarrival + [np.nan]


        # Store features in the object
        self.features = pd.DataFrame({
            'Duration': duration,
            'Severity': severity,
            'Intensity': intensity,
            'Date_Ini_Ev': date_ini_ev,
            'Date_Fin_Ev': date_fin_ev,
            'Interarrival': interarrival,
            'Frequency_of_Occ': frequency,
            'Recovery_Duration': recovery_duration
        })

        return self.features