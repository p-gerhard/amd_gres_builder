# This script assembles the SLURM 'gres.conf' configuration lines for each
# available AMD Instinct GPU. It retrieves necessary information using the
# 'rocm-smi' and 'lscpu' tools.
# Copyright (C) <2024>  <Pierre GERHARD> University of Strasbourg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import logging
import os
import subprocess
from collections import defaultdict
from datetime import datetime

_ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
_RSMI_BIN = f"{_ROCM_PATH}/bin/rocm-smi"


_RSMI_GPU_IDENTIFIER_PATTERN = "card"
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

_RSMI_CARD_SERIE_MAP = {
    "Instinct MI100": "mi100",
    "Instinct MI210": "mi210",
    "Instinct MI250": "mi250",
    "Instinct MI250X": "mi250x",
    "Instinct MI300A": "mi300a",
    "Instinct MI300X": "mi300x",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


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


def _dict_set_keys_to_lower(input):
    """
    Recursively converts all keys in a dictionary to lowercase.

    Args:
        input (dict): The dictionary to process. This can include nested
        dictionaries and lists.

    Returns:
        dict: A new dictionary with all string keys converted to lowercase.
    """

    if isinstance(input, dict):
        new_dict = {}
        for key, value in input.items():
            new_key = key.lower() if isinstance(key, str) else key
            new_dict[new_key] = _dict_set_keys_to_lower(value)
        return new_dict
    elif isinstance(input, list):
        return [_dict_set_keys_to_lower(item) for item in input]
    else:
        return input


def _call_json_cmd(cmd):
    """
    Executes a shell command and parses its JSON output.

    Args:
        cmd (str): The command to be executed.

    Returns:
        dict: The JSON output of the command parsed into a dictionary.
    """
    return json.loads(subprocess.check_output(cmd, shell=True).decode())


def _rocm_smi_get_file():
    """
    Retrieves the device file path of each available GPU by using 'rocm-smi'
    tool.

    Returns:
        dict: A dictionary where keys are GPU identifiers and each associated
        value is a dictionary of this form {'File': '/dev/dri/renderD130'}
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showbus --json")
    rsmi_dict = _dict_set_keys_to_lower(rsmi_dict)

    res = {}
    for key, val in rsmi_dict.items():
        if _RSMI_GPU_IDENTIFIER_PATTERN in key:
            # Getting PCI Bus
            pci_bus = val.get("pci bus")
            assert pci_bus is not None

            # Building device 'by-path' symlink using PCI Bus value
            path = f"/dev/dri/by-path/pci-{pci_bus.lower()}-render"

            # Find device file path through symlink
            path = f"/dev/dri{os.path.abspath(os.readlink(path))}"
            res[key] = {"File": path}

    return res


def _rocm_smi_get_type():
    """
    Retrieves the card serie type of each available GPU by using 'rocm-smi'
    tool. Card serie type are then mapped to short name (eg. "Instinct MI100" ->
    "mi100").

    Returns:
        dict: A dictionary where keys are GPU identifiers and each associated
        value is a dictionary of this form  {'Type': 'mi210'}

    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showproductname --json")
    rsmi_dict = _dict_set_keys_to_lower(rsmi_dict)

    res = {}
    for key, val in rsmi_dict.items():
        # Filter keys by using GPU identifier pattern
        if _RSMI_GPU_IDENTIFIER_PATTERN in key:
            card_serie = val.get("card series")
            assert card_serie is not None

            # Converting value to short name
            card_serie = _RSMI_CARD_SERIE_MAP.get(
                card_serie, card_serie.replace(" ", "_").lower()
            )
            res[key] = {"Type": card_serie}

    return res


def _rocm_smi_get_uuid():
    """
    Retrieves the UUID of each available GPU by using 'rocm-smi' tool.

    Returns:
        dict: A dictionary where keys are GPU identifiers and each associated
        value is a dictionary of this form  {'UUID': '0x9d9d841980e9a221'}
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showuniqueid --json")
    rsmi_dict = _dict_set_keys_to_lower(rsmi_dict)

    res = {}
    for key, val in rsmi_dict.items():
        # Filter keys by using GPU identifier pattern
        if _RSMI_GPU_IDENTIFIER_PATTERN in key:
            uuid = val.get("unique id")
            assert uuid is not None
            res[key] = {"UUID": uuid}

    return res


def _rocm_smi_get_serial():
    """
    Retrieves the serial number of each available GPU by using 'rocm-smi' tool.

    Returns:
        dict: A dictionary where keys are GPU identifiers and each associated
        value is a dictionary of this form  {'UUID': '0x9d9d841980e9a221'}
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showserial --json")

    # Cast all key to lower case
    rsmi_dict = _dict_set_keys_to_lower(rsmi_dict)

    res = {}
    for key, val in rsmi_dict.items():
        # Filter keys by using GPU identifier pattern
        if _RSMI_GPU_IDENTIFIER_PATTERN in key:
            # Getting serial
            serial = val.get("serial number")
            assert serial is not None

            res[key] = {"Serial": serial}

    return res


def _rocm_smi_get_links():
    """
    Retrieves the device to device topology of each available GPU by using
    'rocm-smi' tool.

    Returns:
        dict: A dictionary where keys are GPU identifiers and each associated
        value is a dictionary of this form  {'Links': ['-1', '0', '1', '0',...]}

    Link values:
        -1: link between a GPU and itself
         0: PCIE link between two GPU
         1: XGMI link between two GPU
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showtopo --json")
    rsmi_dict = _dict_set_keys_to_lower(rsmi_dict)

    # Getting GPU count
    nb_gpu = len([k for k in rsmi_dict.keys() if _RSMI_GPU_IDENTIFIER_PATTERN in k])

    # Initializing "Links" to a list where values are all equal to "-1"
    res = {
        f"{_RSMI_GPU_IDENTIFIER_PATTERN}{k}": {"Links": (nb_gpu) * ["-1"]}
        for k in range(nb_gpu)
    }

    # For each value of i and j we try to fetch the json data for a match
    # Warning: rocm-smi only outputs GPU_i -> GPU_j links never the symetrical
    # ones GPU_j -> GPU_i
    tag = "(topology) link type between drm devices {i} and {j}"
    for i in range(nb_gpu):
        for j in range(nb_gpu):
            # Read GPU_i->GPU_j association (if present)
            link_ij = (
                rsmi_dict.get("system")
                .get(tag.format(i=i, j=j), "")
                .replace("PCIE", "0")
                .replace("XGMI", "1")
            )

            if link_ij != "":
                # Store GPU_i->GPU_j link type
                res[f"{_RSMI_GPU_IDENTIFIER_PATTERN}{i}"]["Links"][j] = link_ij
                # Store GPU_j->GPU_i link type
                res[f"{_RSMI_GPU_IDENTIFIER_PATTERN}{j}"]["Links"][i] = link_ij

    return res


def _lscpu_get_numa_cpus():
    """
    Retrieves the cpus of each available NUMA node by using 'lscpu' tool.

    Returns:
        dict: A dictionary where keys are NUMA node IDs and each associated
        value is a string containing the sequence of CPUs '1,2,3,4'. When
        possible, incremental sequence is converted to range eg 1,2,3 -> '1-3'

    """
    numa_data = subprocess.check_output(
        "lscpu --parse=NODE,CPU",
        shell=True,
    ).decode()

    # Storing list of CPUs
    res = defaultdict(list)
    for line in numa_data.split("\n"):
        if not line.startswith("#") and line != "":
            data = line.split(",")
            res[int(data[0])].append(int(data[1]))

    # Converting list of CPUs to range "cpu_min-cpu_max" if possible
    res = dict(res)
    for key, val in res.items():
        cpu_min, cpu_max = min(val), max(val)
        if set(val) == set(range(cpu_min, cpu_max + 1)):
            res[key] = f"{cpu_min}-{cpu_max}"
        else:
            res[key] = ",".join(sorted([str(v) for v in res[key]]))
    return res


def _rocm_smi_get_cores():
    """
    Retrieves associated the cpus core of each available GPU by using 'rocm-smi'
    tool.
    Returns:
        dict: A dictionary where keys are GPU identifiers and each associated
        value is a dictionary of this form  {'Cores': '1,2,3,4'}
        When possible, incremental sequence is converted to range eg '1,2,3' ->
        '1-3'
    """
    rsmi_dict = _call_json_cmd(f"{_RSMI_BIN} --showtoponuma --json")
    rsmi_dict = _dict_set_keys_to_lower(rsmi_dict)

    # Getting NUMA to CPUs map (used to retrieve CPU)
    numa_to_cpus = _lscpu_get_numa_cpus()

    res = {}
    for key, val in rsmi_dict.items():
        # Filter keys by using GPU identifier pattern
        if _RSMI_GPU_IDENTIFIER_PATTERN in key:
            numa_node = val.get("(topology) numa node", "")
            cores = numa_to_cpus.get(int(numa_node))
            res[key] = {"Cores": cores}

    return res


def get_gres_conf():
    """
    Retrieves GPU resource configuration and prints slurm 'gres.conf' lines for
    each GPU.
    """
    serial = _rocm_smi_get_serial()
    uuid = _rocm_smi_get_uuid()
    file = _rocm_smi_get_file()
    links = _rocm_smi_get_links()
    type = _rocm_smi_get_type()
    cores = _rocm_smi_get_cores()

    gres_dict = _merge_dicts(file, links, cores, type, uuid, serial)

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

    # In order to be constant among reboot and rocm versions we sort gathered
    # rocm-smi informations by using serial number.
    gres_dict = dict(sorted(gres_dict.items(), key=lambda item: item[1]["Serial"]))

    # Links value must be rearranged using the same permutation
    for val in gres_dict.values():
        val["Links"] = [
            x
            for _, x in sorted(
                zip([val["Serial"] for val in serial.values()], val["Links"])
            )
        ]

    # Converting "Links" to a sequence string
    for val in gres_dict.values():
        val["Links"] = ",".join(val["Links"])


    # Get the name of the current script
    script_name = os.path.basename( os.path.abspath(__file__))
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{80*'#'}")
    print(f"# AMD GPU 'gres.conf' for host '{os.uname()[1]}'")
    print(f"# Generated by {script_name} on {date}")

    for idx, val in enumerate(gres_dict.values()):
        # Assemble 'gres.conf' line comment
        cmt = (
            f"# GPU {idx} with uuid={val['UUID']} and serial={val['Serial']}"
        )
        print(cmt)

        # Assemble 'gres.conf' line
        line = " ".join(
            [
                f"{subkey}={val[subkey]}"
                for subkey in _GRES_FIELDS_ORDER
                if val.get(subkey) is not None
            ]
        )
        print(line)

    print(f"{80*'#'}")

if __name__ == "__main__":
    # Check if the binary file exists
    if not os.path.exists(_RSMI_BIN):
        print(f"Binary file '{_RSMI_BIN}' does not exist.")
        print(
            "Please set ROCM_PATH environment variable to the proper installation path of ROCm"
        )
        print("Example: export ROCM_PATH=/opt/rocm")
        exit(-1)
    else:
        get_gres_conf()
