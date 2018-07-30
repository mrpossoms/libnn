# This file is part of libnn.
#
# libnn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libnn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libnn.  If not, see <https://www.gnu.org/licenses/>.


def serialize_matrix(m, fp):
    """
    Writes a numpy array into fp in the simple format that
    libnn's nn_mat_load() function understands
    :param m: numpy matrix
    :param fp: file stream
    :return: void
    """
    import struct

    # write the header
    fp.write(struct.pack('b', len(m.shape)))
    for d in m.shape:
        fp.write(struct.pack('i', d))

    # followed by each element
    for e in m.flatten():
        fp.write(struct.pack('f', e))
