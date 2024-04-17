import launch

if not launch.is_installed("numba"):
    launch.run_pip("install numba", "requirements for PRiR") 
if not launch.is_installed("joblib"):
    launch.run_pip("install joblib", "requirements for PRiR")
if not launch.is_installed("seaborn"):
    launch.run_pip("install seaborn", "requirements for PRiR")
if not launch.is_installed("mpi4py"):
    launch.run_pip("install mpi4py", "requirements for PRiR")
if not launch.is_installed("scipy"):
    launch.run_pip("install scipy", "requirements for PRiR")
if not launch.is_installed("cv2"):
    launch.run_pip("install cv2", "requirements for PRiR")