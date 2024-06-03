import json
import os
import subprocess
from collections import defaultdict

_ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
_RSMI_BIN = f"{_ROCM_PATH}/bin/rocm-smi"

_RSMI_GPU_JSON_KEY = "card"
_GRES_FIELDS_ORDER = [
    "NodeName",
    "Name",
    "Type",
    "Autodetect",
    "Count",
    "Cores",
    "Links",
    "Flags",
    "File",
]


def _call_json_cmd(cmd):
    """
    Executes a shell command and parses its JSON output.

    Args:
        cmd (str): The command to be executed.

    Returns:
        dict: The JSON output of the command parsed into a dictionary.
    """
    return json.loads(subprocess.check_output(cmd, shell=True).decode())


def _rsmi_get_gres_file():
    """
    Retrieves GPU resource information using the RSMI tool and constructs device
    file paths.

    Returns:
        dict: A dictionary where keys are GPU identifiers and values are
              dictionaries containing the 'File' key with the corresponding
              device path as its value.
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showbus --json")

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_JSON_KEY in key:
            # Get PCI Bus
            pci_bus = val.get("PCI Bus")
            assert pci_bus is not None

            # Build device 'by-path' symlink using PCI Bus
            path = f"/dev/dri/by-path/pci-{pci_bus.lower()}-render"

            # Find device file path following symlink
            path = f"/dev/dri{os.path.abspath(os.readlink(path))}"
            res[key] = {"File": path}

    return res


def _rsmi_get_gres_type():
    """
    Retrieves GPU type information using the RSMI tool and processes the GFX
    version.

    Returns:
        dict: A dictionary where keys are GPU identifiers and values are
              dictionaries containing the 'Type' key with the processed GFX
              version as its value.
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showproductname --json")

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_JSON_KEY in key:
            # Get GFX version
            gfx_ver = val.get("GFX Version", "").lower()

            # Last two digits of gfx_ver must be cast to hex
            # eg. gfx9010 -> gfx90a
            gfx_ver = f"{gfx_ver[:-2]}{hex(int(gfx_ver[-2:]))[2:]}"
            res[key] = {"Type": gfx_ver}

    return res


def _rsmi_get_gres_links():
    """
    Retrieves GPU topology information using the RSMI tool and constructs the
    link types between GPUs.

    Returns:
        dict: A dictionary where keys are GPU identifiers and values are
              dictionaries containing the 'Links' key with a list of link types
              to other GPUs.
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showtopo --json")

    # Get GPU count
    nb_gpu = len([k for k in rsmi_dict.keys() if _RSMI_GPU_JSON_KEY in k])

    # Initialize result with empty links (filled with 0)
    res = {f"{_RSMI_GPU_JSON_KEY}{k}": {"Links": (nb_gpu) * [0]} for k in range(nb_gpu)}

    tag = "(Topology) Link type between DRM devices {i} and {j}"
    for i in range(nb_gpu):
        for j in range(nb_gpu):
            # Read GPU_i->GPU_j association (if present)
            link_ij = (
                rsmi_dict.get("system")
                .get(tag.format(i=i, j=j), "0")
                .replace("PCIE", "0")
                .replace("XGMI", "1")
            )

            # Read GPU_j->GPU_i association (if present)
            link_ji = (
                rsmi_dict.get("system")
                .get(tag.format(i=j, j=i), "0")
                .replace("PCIE", "0")
                .replace("XGMI", "1")
            )

            res[f"{_RSMI_GPU_JSON_KEY}{i}"]["Links"][j] = str(
                max(int(link_ij), int(link_ji))
            )

            # GPU_i->GPU_i
            if i == j:
                res[f"{_RSMI_GPU_JSON_KEY}{i}"]["Links"][j] = "-1"

    # Merge list element into a comme separated string
    for val in res.values():
        val = ",".join(val)

    return res


def _lscpu_get_numa_cpus():
    """
    Retrieves CPU information per NUMA node using lscpu command.

    Returns:
        dict: A dictionary where keys are NUMA node IDs and values are CPU
        ranges or lists.
    """
    numa_data = subprocess.check_output(
        "lscpu --parse=NODE,CPU",
        shell=True,
    ).decode()

    # For each NUMA node, store the list of CPUs
    res = defaultdict(list)
    for line in numa_data.split("\n"):
        if not line.startswith("#") and line != "":
            data = line.split(",")
            res[int(data[0])].append(int(data[1]))

    # Convert list of CPUs to range "cpu_min-cpu_max" if possible
    res = dict(res)
    for key, val in res.items():
        cpu_min, cpu_max = min(val), max(val)
        if set(val) == set(range(cpu_min, cpu_max + 1)):
            res[key] = f"{cpu_min}-{cpu_max}"

    return res


def _rsmi_get_gres_cores():
    """
    Retrieves GPU core information per NUMA node using the RSMI tool.

    Returns:
        dict: A dictionary where keys are GPU identifiers and values are
              dictionaries containing the 'Cores' key with CPU cores associated
              with the GPU.
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showtoponuma --json")

    # Get NUMA to CPUs map
    numa_to_cpus = _lscpu_get_numa_cpus()

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_JSON_KEY in key:
            numa_node = val.get("(Topology) Numa Node", "")
            cores = numa_to_cpus.get(int(numa_node))
            res[key] = {"Cores": cores}

    return res


def _merge_dicts(*dicts):
    """
    Merges multiple dictionaries into a single dictionary by combining values of
    corresponding keys.

    Args:
        *dicts: Variable number of dictionaries to merge.

    Returns:
        dict: Merged dictionary with values combined from corresponding keys
        across input dictionaries.
    """
    keys = dicts[0].keys()
    return {key: {k: v for d in dicts for k, v in d[key].items()} for key in keys}


def get_gres_conf():
    """
    Retrieves GPU resource configuration and prints slurm 'gres.conf' lines for
    each GPU.
    """

    file = _rsmi_get_gres_file()
    links = _rsmi_get_gres_links()
    type = _rsmi_get_gres_type()
    cores = _rsmi_get_gres_cores()

    gres_dict = _merge_dicts(file, links, cores, type)

    # Add constant 'gres.conf' fields to each key of gres_dict
    hostname = os.uname()[1].split(".")[0]
    for key in gres_dict:
        gres_dict[key].update(
            {
                "Name": "gpu",
                "NodeName": hostname,
                "Autodetect": "off",
                "Count": 1,
                "Flags": "amd_gpu_env",
            }
        )

    # Build 'gres.conf' line for each GPU
    for key, val in gres_dict.items():
        line = " ".join(
            [
                f"{subkey}={val[subkey]}"
                for subkey in _GRES_FIELDS_ORDER
                if val.get(subkey) is not None
            ]
        )
        print(line)


if __name__ == "__main__":
    # Check if the binary file exists
    if not os.path.exists(_RSMI_BIN):
        print(f"Binary file '{_RSMI_BIN}' does not exist.")
        print(
            "Please set ROCM_PATH environment variable to the proper installation path of ROCm eg."
        )
        print("export ROCM_PATH=/opt/rocm")
        exit(-1)
    else:
        get_gres_conf()
