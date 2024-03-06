import launch

if not launch.is_installed("numba"):
    launch.run_pip("install numba", "requirements for PRiR") 
if not launch.is_installed("joblib"):
    launch.run_pip("install joblib", "requirements for PRiR")