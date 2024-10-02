# core function:
# * takes a filepath and params as input filepath = Part of the params object
# * read the file
# * converts the file to a standard format
# * applies the analysis algorithm
# * returns the result


from digatron_utility.read import DigatronDataFormat  # , read_digatron_data_file
from maccor_utility.read import MaccorDataFormat  # , read_maccor_data_file

import aelcha.process_maccor_data_legacy as pmd
from aelcha.user_interface import (  # FileSelection,
    Configuration,
    LimitOption,
    SelectionRow,
)


def process_file(row: SelectionRow, config: Configuration):
    if (
        config.input_source_type.name in MaccorDataFormat.__members__
        or config.input_source_type.name in DigatronDataFormat.__members__
    ):
        # maccor_data = read_maccor_data_file(
        #     file_path=row._file_path,
        #     frmt=MaccorDataFormat[config.input_source_type.name]
        # )
        # Adapter
        # * read only certain columns from the data into the unified structure
        #   * Mapping of columns from the original data to the unified structure
        #   * Use DataFrame representation
        #       * rename columns
        #       * drop columns
        #       * apply factors to columns
        #       * reformat columns
        #   * Cast the data and metadata into the unified structure

        row_dict = row.model_dump()
        config_dict = config.model_dump()
        params = {}
        # get the names of the arguments of pmd.process_maccor_data_legacy
        pmd_args = pmd.process_maccor_data.__code__.co_varnames
        for key in row_dict:
            val = row_dict[key]
            if hasattr(val, "value"):
                val = val.value
            if isinstance(val, dict):
                if "numerical_value" in val:
                    val = val["numerical_value"]
            if key in pmd_args:
                params[key] = val
        for key in config_dict:
            val = config_dict[key]
            if hasattr(val, "value"):
                val = val.value
            if key in pmd_args:
                params[key] = val

        def process_limits(limit1, limit2):
            if limit1 or limit2 == LimitOption.auto:
                return LimitOption.auto.name
            return limit1, limit2

        for key, lim1, lim2 in [
            (
                "v_c_x_limits",
                config.v_vs_cap_2el_x_low_lim,
                config.v_vs_cap_2el_x_up_lim,
            ),
            (
                "v_c_y_limits",
                config.v_vs_cap_2el_y_low_lim,
                config.v_vs_cap_2el_y_up_lim,
            ),
            (
                "v_c_y1_limits",
                config.v_vs_cap_3el_left_y_low_lim,
                config.v_vs_cap_3el_left_y_up_lim,
            ),
            (
                "v_c_y2_limits",
                config.v_vs_cap_3el_right_y_low_lim,
                config.v_vs_cap_3el_right_y_up_lim,
            ),
            (
                "dq_dv_x_limits",
                config.dq_dv_vs_v_2el_x_low_lim,
                config.dq_dv_vs_v_2el_x_up_lim,
            ),
            (
                "dq_dv_y_limits",
                config.dq_dv_vs_v_2el_y_low_lim,
                config.dq_dv_vs_v_2el_y_up_lim,
            ),
            (
                "dq_dv_x1_limits",
                config.dq_dv_vs_v_3el_left_x_low_lim,
                config.dq_dv_vs_v_3el_left_x_up_lim,
            ),
            (
                "dq_dv_x2_limits",
                config.dq_dv_vs_v_3el_right_x_low_lim,
                config.dq_dv_vs_v_3el_right_x_up_lim,
            ),
        ]:
            params[key] = process_limits(lim1, lim2)

        pmd.process_maccor_data(**params)

    # elif config.input_source_type.name in DigatronDataFormat.__members__:
    #     digatron_data = read_digatron_data_file(
    #         file_path=row._file_path,
    #         frmt=DigatronDataFormat[config.input_source_type.name]
    #     )
    #     # Adapter
