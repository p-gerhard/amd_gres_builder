import json
import os
import subprocess
from collections import defaultdict

_RSMI_BIN = "/opt/rocm/bin/rocm-smi"
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
    return json.loads(subprocess.check_output(cmd, shell=True).decode())


def _rsmi_get_gres_file():
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showbus --json")

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_JSON_KEY in key:
            # Get PCI Bus
            pci_bus = val.get("PCI Bus")
            assert pci_bus is not None

            # Build device 'by-path' symlink using PCI Bus
            path = f"/dev/dri/by-path/pci-{pci_bus.lower()}-render"

            # Find device path following symlink
            path = f"/dev/dri{os.path.abspath(os.readlink(path))}"
            res[key] = {"File": path}

    return res


def _rsmi_get_gres_type():
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showproductname --json")

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_JSON_KEY in key:
            # Get GFX version
            gfx_ver = val.get("GFX Version", "").lower()

            # Last two digits of gfx_ver must be casted to hex
            # eg. gfx9010 -> gfx90a
            gfx_ver = f"{gfx_ver[:-2]}{hex(int(gfx_ver[-2:]))[2:]}"
            res[key] = {"Type": gfx_ver}

    return res


def _rsmi_get_gres_links():
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showtopo --json")

    # Get GPU number
    nb_gpu = len([k for k in rsmi_dict.keys() if _RSMI_GPU_JSON_KEY in k])

    # Set empty links (filled with 0)
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

    return res


def _lscpu_get_numa_cpus():
    numa_data = subprocess.check_output(
        "lscpu --parse=NODE,CPU",
        shell=True,
    ).decode()

    # For each numa node store the list of cpus
    res = defaultdict(list)
    for line in numa_data.split("\n"):
        if not line.startswith("#") and line != "":
            data = line.split(",")
            res[int(data[0])].append(int(data[1]))

    # Convert list of cpus to range "cpu_min-cpu_max" if possible
    res = dict(res)
    for key, val in res.items():
        cpu_min, cpu_max = min(val), max(val)
        if set(val) == set(range(cpu_min, cpu_max + 1)):
            res[key] = f"{cpu_min}-{cpu_max}"

    return res


def _rsmi_get_gres_cores():
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showtoponuma --json")

    # Get numa to cpus map
    numa_to_cpus = _lscpu_get_numa_cpus()

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_JSON_KEY in key:
            numa_node = val.get("(Topology) Numa Node", "")
            cores = numa_to_cpus.get(int(numa_node))
            res[key] = {"Cores": cores}

    return res


def _merge_dicts(*dicts):
    keys = dicts[0].keys()
    return {key: {k: v for d in dicts for k, v in d[key].items()} for key in keys}


def _add_gres_constant_fields(dict, dict_cst):
    for key in dict:
        dict[key].update(dict_cst)


def get_gres_conf():
    file = _rsmi_get_gres_file()
    links = _rsmi_get_gres_links()
    type = _rsmi_get_gres_type()
    cores = _rsmi_get_gres_cores()

    dict = _merge_dicts(file, links, cores, type)

    # Add constant fields to each key of dict
    for key in dict:
        dict[key].update(
            {
                "Name": "gpu",
                "NodeName": os.uname()[1].split(".")[0],
                "Autodetect": "off",
                "Count": 1,
                "Flags": "amd_gpu_env",
            }
        )

    # Build gres.conf line for each gpu
    for key, val in dict.items():
        line = ""
        for subkey in _GRES_FIELDS_ORDER:
            res = val.get(subkey)

            if res is not None:
                if subkey == "Links":
                    line += f"{subkey}={','.join(res)} "
                else:
                    line += f"{subkey}={res} "
        print(line)


if __name__ == "__main__":
    get_gres_conf()
