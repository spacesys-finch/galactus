# GALACTUS THE ALL-KNOWING
# Sensitivity analysis for our orbit

from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
import numpy as np

import os

from numba import njit
from poliastro.core.perturbations import (J2_perturbation,
                                          atmospheric_drag_exponential)
from poliastro.constants import rho0_earth, H0_earth


from poliastro.twobody.propagation import cowell, propagate

from scipy.interpolate import interp1d

from celest import Satellite, GroundPosition, Encounter

for SMA in np.linspace(6878, 6928, 20):
    for LTANP in [22.5]:
        for Eccentricity in np.linspace(0, 0.01, 20):
            print(f"SMA = {SMA}, LTAN = {LTANP}, ECC = {Eccentricity}")
            argument_of_perigee = 0
            total_anomaly = 0
            epoch_date = "2023-06-01"
            epoch_time = "00:00"

            # convert parameters to dimensionful values
            a = SMA * u.km
            e = Eccentricity * u.one
            argpe = argument_of_perigee * u.deg
            ta = total_anomaly * u.deg
            LTAN = LTANP % 24 * u.hourangle

            # Store time as UTC
            epoch = Time(
                f"{epoch_date} {epoch_time}",
                format='iso',
                scale="utc")

            # Define orbital parameters, along with units
            orb = Orbit.heliosynchronous(
                attractor=Earth,
                a=a,
                ecc=e,
                ltan=LTAN,
                argp=argpe,
                nu=ta,
                epoch=epoch)

            print(orb)
            # Time in orbit is now stored as TDB

            print("Inclination:", orb.inc.value * 180 / np.pi)
            print("Period:", orb.period)

            C_D = 2.2
            FRONTAL_AREA = 0.03
            MASS = 3
            A_over_m = (FRONTAL_AREA * u.m**2 / (MASS * u.kg)
                        ).to_value(u.km ** 2 / u.kg)  # km^2/kg

            ###################################################################
            @njit
            def a_d(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
                return J2_perturbation(t0, state, k, J2, R) + \
                    atmospheric_drag_exponential(
                        t0, state, k, R, C_D, A_over_m, H0, rho0)

            # Force model
            def f(t0, state, k):
                return a_d(t0, state, k, R=Earth.R.to(u.km).value, C_D=C_D,
                           A_over_m=A_over_m,
                           H0=H0_earth.to(u.km).value,
                           rho0=rho0_earth.to(u.kg / u.km ** 3).value,
                           J2=Earth.J2.value)

            ###################################################################
            num_days = 50
            num_steps = 50000

            tof = (num_days * u.d).to(u.s)  # time in seconds

            # Mission elapsed timesteps
            time_MET = np.linspace(0, tof, num_steps)

            # Obtain propagated positions and velocities using Cowell's method
            rr = propagate(orb, time_MET, method=cowell, ad=f)
            vv = rr.differentials["s"]

            # Obtain jd dates along the propagation steps to feed into astropy as position data
            # We work in TDB for propagation since the orbit data is stored in
            # TDB
            time_absolute = Time(
                np.linspace(
                    0,
                    num_days,
                    num_steps) +
                orb.epoch.jd,
                format="jd",
                scale="tdb")
            time_absolute.format = "iso"

            print(
                f"Cowell Propagation successful to {time_absolute[-1].utc} UTC")

            ECI_interp = interp1d(time_MET, rr.xyz, kind="cubic")

            fine_MET = np.linspace(0, tof, num_steps * 10)

            fine_absolute = Time(
                np.linspace(
                    0,
                    num_days,
                    num_steps *
                    10) +
                orb.epoch.utc.jd,
                format="jd",
                scale="utc")
            fine_absolute.format = "iso"

            UTCTimeData = fine_absolute.value.astype(str)
            # This hangs up for large arrays

            toronto = GroundPosition(
                name="Toronto", coor=(
                    43.662300, -79.394530))
            east_trout_lake = GroundPosition(
                name="East Trout Lake", coor=(
                    54.36762, -105.08050))

            finch = Satellite()

            NdrAng = finch.getNdrAng(
                groundPos=toronto,
                posData=ECI_interp(fine_MET).T,
                timeData=UTCTimeData)
            Alt, Az = finch.getAltAz(
                groundPos=toronto, posData=ECI_interp(fine_MET).T, timeData=UTCTimeData)

            finch.getNdrAng(
                groundPos=east_trout_lake,
                posData=ECI_interp(fine_MET).T,
                timeData=UTCTimeData)
            finch.getAltAz(
                groundPos=east_trout_lake,
                posData=ECI_interp(fine_MET).T,
                timeData=UTCTimeData)

            encounters = Encounter()
            encounters.addEncounter(
                "CYYZ IMG",
                "IMG",
                toronto,
                30,
                "nadirLOS",
                True,
                solar=1)
            encounters.addEncounter(
                "TROUT LAKE CALIBRATION",
                "IMG",
                east_trout_lake,
                30,
                "nadirLOS",
                True,
                solar=1)
            encounters.addEncounter(
                "CYYZ DL", "DL", toronto, 10, "alt", False, solar=-1)

            encounters.getWindows(finch)

            RESULT_NAME = f"results/SMA{SMA}LTAN{LTANP}ECC{Eccentricity}AOP{argument_of_perigee}".replace(
                ".", "")

            os.mkdir(f"./{RESULT_NAME}")

            try:
                encounters.getStats().to_csv(
                    f"./{RESULT_NAME}/encounter_stats.csv")
            except BaseException as e:
                print(e)

            try:
                encounters.saveWindows(f"./{RESULT_NAME}/encounters.txt", "\t")
            except BaseException as e:
                print(e)

            np.save(f"./{RESULT_NAME}/ECI_positions.npy", rr.xyz.T.value)
            np.save(f"./{RESULT_NAME}/timesteps.npy", time_MET.value)
            np.save(f"./{RESULT_NAME}/time_jd.npy", time_absolute.jd)

            print(f"Done {RESULT_NAME}")
