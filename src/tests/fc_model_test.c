/*
 * This file is part of libnn.
 *
 * libnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libnn.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "test.h"
#include "../nn.h"
#include "../nn.c"

float hay[] = {0.26862746,0.100000024,-0.08823529,0.45294118,0.2529412,0.03725493,0.3392157,0.15490198,-0.049019605,0.19803923,0.03725493,-0.15098038,0.327451,0.15098041,-0.049019605,0.38235295,0.2019608,0.001960814,0.44117647,0.24901962,0.049019635,0.49607843,0.29607844,0.09215689,0.46470588,0.2764706,0.07647061,0.30392158,0.1156863,-0.08039215,0.5,0.38235295,0.15490198,0.25686276,0.10392159,-0.08039215,0.27254903,0.08823532,-0.123529404,0.37058824,0.17058825,-0.05686274,0.33529413,0.15882355,-0.06470588,0.25686276,0.072549045,-0.14313725,0.39411765,0.19411767,-0.017647058,0.43333334,0.23725492,0.045098066,0.37843138,0.18627453,0.0058823824,0.4372549,0.2137255,0.025490224,0.28431374,0.08039218,-0.08823529,0.33529413,0.13137257,-0.049019605,0.31176472,0.119607866,-0.049019605,0.29607844,0.13529414,-0.013725489,0.20588237,0.072549045,-0.06470588,0.27254903,0.13921571,0.0058823824,0.41764706,0.27254903,0.13529414,0.37058824,0.2019608,0.025490224,0.37843138,0.1901961,-0.009803921,0.38235295,0.19411767,-0.009803921,0.47254902,0.2764706,0.072549045,0.3156863,0.14313728,-0.029411763,0.29215688,0.127451,-0.06470588,0.41764706,0.23725492,0.03333336,0.18235296,0.029411793,-0.14313725,0.16274512,-0.0019607842,-0.19411764,0.45686275,0.25686276,0.03333336,0.42941177,0.23725492,0.021568656,0.3156863,0.14313728,-0.045098037,0.26862746,0.100000024,-0.06862745,0.46862745,0.2647059,0.06862748,0.45294118,0.25686276,0.06078434,0.45686275,0.28431374,0.07647061,0.21764708,0.052941203,-0.13529411,0.37843138,0.18235296,-0.045098037,0.38235295,0.17843139,-0.052941173,0.34705883,0.14705884,-0.08039215,0.36666667,0.17450982,-0.049019605,0.3627451,0.17058825,-0.0372549,0.48431373,0.29215688,0.08039218,0.3627451,0.16274512,-0.025490195,0.35882354,0.15490198,-0.029411763,0.42156863,0.20980394,0.013725519,0.34705883,0.14313728,-0.049019605,0.39411765,0.2019608,0.0058823824,0.2019608,0.049019635,-0.10784313,0.20980394,0.06862748,-0.072549015,0.24901962,0.127451,0.0058823824,0.39803922,0.25686276,0.1156863,0.30392158,0.15490198,-0.0058823526,0.22549021,0.052941203,-0.14705881,0.3,0.119607866,-0.08039215,0.33529413,0.15098041,-0.045098037,0.40980393,0.22549021,0.03725493,0.39803922,0.23333335,0.013725519,0.31176472,0.15882355,-0.025490195,0.31176472,0.13529414,-0.06078431,0.3509804,0.15882355,-0.045098037,0.3627451,0.17450982,-0.03333333,0.40588236,0.20588237,-0.009803921,0.31176472,0.127451,-0.07647058,0.3392157,0.15098041,-0.049019605,0.45686275,0.25686276,0.06470591,0.5,0.29215688,0.10392159,0.49607843,0.31176472,0.10784316,0.3392157,0.15882355,-0.045098037,0.45686275,0.26078433,0.025490224,0.35490197,0.15882355,-0.06470588,0.24901962,0.06078434,-0.15098038,0.19411767,0.013725519,-0.19411764,0.2019608,0.03333336,-0.15882352,0.37058824,0.18235296,-0.013725489,0.3,0.11176473,-0.06470588,0.23725492,0.05686277,-0.099999994,0.45294118,0.24509805,0.05686277,0.39019608,0.18235296,-0.013725489,0.43333334,0.23333335,0.021568656,0.27254903,0.09215689,-0.09215686,0.35882354,0.17450982,-0.009803921,0.2882353,0.127451,-0.045098037,0.45294118,0.2882353,0.10392159,0.2882353,0.123529434,-0.06078431,0.26078433,0.08823532,-0.10784313,0.327451,0.14705884,-0.05686274,0.28431374,0.10784316,-0.08431372,0.31960785,0.14705884,-0.041176468,0.35882354,0.18627453,-0.017647058,0.3,0.13921571,-0.029411763,0.3392157,0.14705884,-0.049019605,0.40588236,0.2019608,-0.013725489,0.3745098,0.16666669,-0.049019605,0.4137255,0.20588237,-0.017647058,0.46078432,0.24509805,0.017647088,0.46078432,0.24901962,0.03725493,0.41764706,0.22549021,0.045098066,0.3627451,0.18627453,0.017647088,0.42156863,0.24509805,0.05686277,0.37058824,0.19411767,-0.009803921,0.32352942,0.15882355,-0.041176468,0.31176472,0.13137257,-0.08039215,0.30392158,0.11176473,-0.10784313,0.30392158,0.10784316,-0.1117647,0.28039217,0.10392159,-0.10392156,0.31176472,0.127451,-0.05686274,0.2764706,0.100000024,-0.05686274,0.25686276,0.08431375,-0.06862745,0.3745098,0.18235296,0.009803951,0.2882353,0.10784316,-0.06470588,0.33137256,0.13137257,-0.052941173,0.39411765,0.18627453,-0.0058823526,0.35490197,0.15882355,-0.021568626,0.36666667,0.18235296,0.0058823824,0.34705883,0.18235296,0.0058823824,0.34705883,0.17450982,-0.021568626,0.3745098,0.2019608,0.0058823824,0.40980393,0.21764708,0.009803951,0.3156863,0.13137257,-0.072549015,0.27254903,0.100000024,-0.08823529,0.46078432,0.3,0.1156863,0.23333335,0.08431375,-0.06862745,0.13137257,-0.013725489,-0.15882352,0.1901961,0.013725519,-0.16666666,0.1901961,0.013725519,-0.1745098,0.26862746,0.08431375,-0.11568627,0.24901962,0.06862748,-0.12745097,0.13529414,-0.017647058,-0.17843136,0.13137257,-0.017647058,-0.16666666,0.5,0.32352942,0.127451,0.39803922,0.22549021,0.03725493,0.22156864,0.06862748,-0.11568627,0.26862746,0.11176473,-0.08823529,0.327451,0.14705884,-0.06078431,0.37843138,0.18235296,-0.03333333,0.3745098,0.17450982,-0.0372549,0.36666667,0.16666669,-0.029411763,0.22156864,0.06470591,-0.10392156,0.2137255,0.06470591,-0.08823529,0.2529412,0.09215689,-0.06078431,0.26078433,0.096078455,-0.06470588,0.23725492,0.06862748,-0.08823529,0.21764708,0.041176498,-0.11568627,0.06470591,-0.08431372,-0.21764705,0.28039217,0.096078455,-0.06862745,0.4372549,0.24117649,0.06862748,0.4254902,0.2529412,0.072549045,0.48431373,0.30784315,0.119607866,0.38235295,0.2137255,0.03333336,0.33137256,0.15882355,-0.017647058,0.32352942,0.15098041,-0.025490195,0.35490197,0.15882355,-0.0372549,0.4137255,0.24509805,0.07647061,0.14313728,-0.0019607842,-0.14313725,0.1156863,-0.0372549,-0.1980392,0.19803923,0.0058823824,-0.18235293,0.22941178,0.017647088,-0.20196077,0.20588237,0.0058823824,-0.20196077,0.24901962,0.05686277,-0.14705881,0.15098041,-0.013725489,-0.19019607,0.19411767,0.041176498,-0.123529404,0.44117647,0.2764706,0.08823532,0.38235295,0.2137255,0.029411793,0.24509805,0.08823532,-0.096078426,0.2764706,0.11176473,-0.08039215,0.123529434,-0.025490195,-0.1980392,0.10392159,-0.049019605,-0.22156861,0.30784315,0.1156863,-0.07647058,0.35882354,0.15882355,-0.025490195,0.15490198,0.029411793,-0.13529411,0.2137255,0.07647061,-0.08431372,0.25686276,0.1156863,-0.049019605,0.31176472,0.16666669,-0.021568626,0.31176472,0.15098041,-0.03333333,0.34313726,0.15490198,-0.025490195,0.23725492,0.06470591,-0.1117647,0.3509804,0.15490198,-0.013725489,0.4764706,0.28039217,0.10392159,0.4764706,0.29215688,0.1156863,0.41764706,0.24509805,0.06862748,0.37058824,0.20980394,0.041176498,0.45686275,0.3,0.119607866,0.33529413,0.16274512,-0.0019607842,0.327451,0.127451,-0.05686274,0.35490197,0.17843139,0.029411793,0.123529434,-0.029411763,-0.17058823,0.17450982,-0.009803921,-0.19411764,0.28039217,0.072549045,-0.13137254,0.4490196,0.2019608,-0.041176468,0.3,0.08823532,-0.14313725,0.2529412,0.05686277,-0.15882352,0.14705884,-0.025490195,-0.20980391,0.22549021,0.06862748,-0.096078426,0.35882354,0.1901961,0.013725519,0.40588236,0.24117649,0.06078434,0.20980394,0.05686277,-0.11568627,0.1901961,0.049019635,-0.12745097,0.18235296,0.03333336,-0.15098038,0.28039217,0.100000024,-0.09215686,0.40588236,0.20588237,0.009803951,0.3745098,0.17843139,0.0058823824,0.32352942,0.14705884,-0.017647058,0.3745098,0.19803923,0.017647088,0.34705883,0.18235296,-0.009803921,0.41764706,0.24901962,0.03333336,0.3392157,0.17450982,-0.029411763,0.24901962,0.07647061,-0.11568627,0.24901962,0.072549045,-0.11568627,0.31176472,0.123529434,-0.05686274,0.39803922,0.20588237,0.029411793,0.3392157,0.15882355,-0.0058823526,0.31176472,0.14313728,-0.025490195,0.29215688,0.127451,-0.03333333,0.39803922,0.22156864,0.052941203,0.32352942,0.15098041,-0.013725489,0.26078433,0.072549045,-0.09215686,0.36666667,0.18235296,0.03333336,0.22941178,0.049019635,-0.1117647,0.2647059,0.06078434,-0.13137254,0.39019608,0.16274512,-0.052941173,0.38627452,0.15490198,-0.072549015,0.2137255,0.0058823824,-0.20980391,0.20980394,0.009803951,-0.20588234,0.2529412,0.06470591,-0.13529411,0.2882353,0.119607866,-0.06862745,0.3156863,0.15882355,-0.013725489,0.49607843,0.31960785,0.13137257,0.39803922,0.22156864,0.03725493,0.23725492,0.07647061,-0.10784313,0.28039217,0.10392159,-0.08431372,0.42156863,0.2137255,0.0058823824,0.43333334,0.22941178,0.03725493,0.36666667,0.17843139,0.0058823824,0.28431374,0.119607866,-0.045098037,0.127451,0.001960814,-0.14705881,0.14705884,-0.0019607842,-0.16666666,0.34705883,0.16666669,-0.0372549,0.3,0.123529434,-0.08823529,0.23725492,0.05686277,-0.14313725,0.23725492,0.06078434,-0.13529411,0.3,0.1156863,-0.07647058,0.3156863,0.13137257,-0.052941173,0.3156863,0.13921571,-0.0372549,0.34705883,0.16666669,-0.0058823526,0.32352942,0.15882355,-0.009803921,0.40980393,0.22941178,0.06078434,0.39411765,0.21764708,0.052941203,0.36666667,0.18627453,0.021568656,0.42941177,0.24117649,0.06078434,0.42156863,0.21764708,0.013725519,0.36666667,0.14705884,-0.06862745,0.39019608,0.15490198,-0.07647058,0.37843138,0.14313728,-0.08431372,0.18235296,-0.017647058,-0.22156861,0.2019608,0.009803951,-0.19411764,0.096078455,-0.05686274,-0.22549018,0.10392159,-0.029411763,-0.1862745,0.20980394,0.08039218,-0.07647058,0.5,0.37843138,0.1901961,0.39411765,0.21764708,0.041176498,0.39411765,0.20980394,0.017647088,0.4137255,0.2137255,0.0058823824,0.30784315,0.123529434,-0.06078431,0.27254903,0.09215689,-0.08039215,0.2019608,0.03725493,-0.123529404,0.14705884,-0.0058823526,-0.15098038,0.096078455,-0.041176468,-0.17843136,0.13529414,-0.029411763,-0.18235293,0.21764708,0.03725493,-0.14313725,0.17843139,-0.0058823526,-0.1862745,0.22156864,0.029411793,-0.15490195,0.22156864,0.03725493,-0.14313725,0.20588237,0.041176498,-0.13921568,0.20980394,0.052941203,-0.11568627,0.22156864,0.06078434,-0.099999994,0.30784315,0.127451,-0.041176468,0.33137256,0.14313728,-0.03333333,0.33529413,0.14705884,-0.029411763,0.42941177,0.23725492,0.05686277,0.5,0.32352942,0.14313728,0.4490196,0.26078433,0.06862748,0.45686275,0.24509805,0.021568656,0.38627452,0.17058825,-0.06078431,0.3509804,0.119607866,-0.1117647,0.3627451,0.127451,-0.08823529,0.23333335,0.03725493,-0.13921568,0.17450982,-0.017647058,-0.1980392,0.100000024,-0.06078431,-0.23333332,0.127451,-0.013725489,-0.18235293,0.23725492,0.100000024,-0.07647058,0.49607843,0.31960785,0.127451,0.26078433,0.072549045,-0.096078426,0.44117647,0.22156864,0.017647088,0.40588236,0.20588237,-0.009803921,0.18627453,0.017647088,-0.17058823,0.2647059,0.100000024,-0.099999994,0.32352942,0.15882355,-0.045098037,0.23725492,0.08039218,-0.096078426,0.21764708,0.03725493,-0.11568627,0.16666669,-0.0058823526,-0.16274509,0.23725492,0.045098066,-0.1117647,0.23333335,0.03725493,-0.123529404,0.2137255,0.017647088,-0.13921568,0.19803923,0.017647088,-0.13137254,0.2647059,0.07647061,-0.06862745,0.24117649,0.06862748,-0.072549015,0.09215689,-0.045098037,-0.1862745,0.20980394,0.025490224,-0.14313725,0.22549021,0.03725493,-0.14705881,0.38235295,0.18627453,0.001960814,0.4254902,0.22941178,0.045098066,0.39019608,0.2137255,0.045098066,0.42941177,0.24117649,0.052941203,0.4019608,0.2137255,0.013725519,0.39803922,0.19411767,-0.029411763,0.35490197,0.13529414,-0.096078426,0.30784315,0.08039218,-0.12745097,0.2882353,0.06862748,-0.10784313,0.3,0.09215689,-0.08431372,0.25686276,0.08039218,-0.099999994,0.24901962,0.08823532,-0.10784313,0.3,0.14705884,-0.052941173,0.49607843,0.3,0.096078455,0.39019608,0.17058825,-0.021568626,0.37058824,0.14705884,-0.05686274,0.40980393,0.21764708,-0.013725489,0.40588236,0.22941178,-0.013725489,0.46078432,0.28431374,0.025490224,0.46862745,0.2882353,0.041176498,0.42941177,0.24117649,0.017647088,0.32352942,0.127451,-0.072549015,0.37058824,0.15882355,-0.029411763,0.3745098,0.15882355,-0.03333333,0.34705883,0.127451,-0.049019605,0.2647059,0.05686277,-0.096078426,0.2019608,0.017647088,-0.119607836,0.123529434,-0.0372549,-0.15882352,0.045098066,-0.08823529,-0.20588234,0.03725493,-0.096078426,-0.1980392,0.35882354,0.15490198,-0.03333333,0.21764708,0.03333336,-0.14705881,0.3627451,0.17058825,-0.013725489,0.26078433,0.09215689,-0.07647058,0.19411767,0.049019635,-0.099999994,0.22941178,0.07647061,-0.07647058,0.30784315,0.14705884,-0.03333333,0.38627452,0.19803923,-0.017647058,0.39019608,0.17450982,-0.052941173,0.327451,0.10392159,-0.099999994,0.19411767,-0.013725489,-0.1862745,0.24117649,0.049019635,-0.12745097,0.36666667,0.17058825,-0.025490195,0.35490197,0.17058825,-0.045098037,0.38627452,0.19803923,-0.021568626,0.4137255,0.2019608,0.001960814,0.33529413,0.11176473,-0.07647058,0.32352942,0.100000024,-0.099999994,0.42941177,0.22941178,-0.009803921,0.46078432,0.27254903,0.025490224,0.49607843,0.31176472,0.06078434,0.46862745,0.2764706,0.045098066,0.4764706,0.2647059,0.045098066,0.3392157,0.13921571,-0.052941173,0.3156863,0.10392159,-0.08431372,0.3,0.100000024,-0.099999994,0.3509804,0.13137257,-0.06470588,0.327451,0.1156863,-0.06078431,0.3392157,0.13921571,-0.0372549,0.21764708,0.052941203,-0.10392156,0.18235296,0.03725493,-0.10784313,0.30392158,0.14313728,-0.009803921,0.327451,0.127451,-0.06470588,0.29607844,0.10784316,-0.072549015,0.3156863,0.14313728,-0.021568626,0.30784315,0.15098041,-0.0019607842,0.28431374,0.13137257,-0.013725489,0.35490197,0.19411767,0.029411793,0.36666667,0.2019608,0.017647088,0.35490197,0.17843139,-0.03333333,0.35882354,0.15490198,-0.06470588,0.39411765,0.17058825,-0.0372549,0.3156863,0.100000024,-0.08039215,0.2764706,0.07647061,-0.099999994,0.38627452,0.18235296,-0.021568626,0.35882354,0.17058825,-0.05686274,0.45686275,0.24509805,0.017647088,0.31176472,0.09215689,-0.10784313,0.32352942,0.096078455,-0.09215686,0.28039217,0.06470591,-0.13137254,0.22941178,0.041176498,-0.15098038,0.22549021,0.06470591,-0.123529404,0.3745098,0.2019608,-0.0058823526,0.45686275,0.25686276,0.045098066,0.4764706,0.24901962,0.052941203,0.3627451,0.15098041,-0.045098037,0.26862746,0.06470591,-0.12745097,0.20588237,0.017647088,-0.1745098,0.33137256,0.123529434,-0.06470588,0.31176472,0.11176473,-0.08039215,0.26078433,0.07647061,-0.10784313,0.25686276,0.08039218,-0.09215686,0.27254903,0.11176473,-0.049019605,0.31960785,0.14313728,-0.025490195,0.27254903,0.08823532,-0.08039215,0.29607844,0.1156863,-0.045098037,0.327451,0.15490198,0.0058823824,0.3,0.13921571,-0.0019607842,0.1901961,0.052941203,-0.08431372,0.34313726,0.18235296,0.03333336,0.4490196,0.28039217,0.096078455,0.5,0.3392157,0.1156863,0.28431374,0.10784316,-0.096078426,0.35882354,0.15490198,-0.041176468,0.4137255,0.19803923,0.0058823824,0.3745098,0.16274512,-0.021568626,0.40980393,0.20588237,-0.0019607842,0.39803922,0.19411767,-0.0372549,0.45294118,0.22549021,-0.009803921,0.2019608,-0.017647058,-0.21372548,0.327451,0.09215689,-0.099999994,0.13137257,-0.072549015,-0.24117646,0.127451,-0.049019605,-0.21372548,0.22156864,0.052941203,-0.119607836,0.34313726,0.16666669,-0.021568626,0.33529413,0.14313728,-0.029411763,0.45294118,0.22549021,0.045098066,0.35490197,0.15098041,-0.041176468,0.24901962,0.052941203,-0.13921568,0.2137255,0.013725519,-0.18235293,0.2882353,0.08823532,-0.1117647,0.31176472,0.11176473,-0.08823529,0.26078433,0.072549045,-0.119607836,0.33529413,0.14705884,-0.049019605,0.33137256,0.15882355,-0.021568626,0.27254903,0.1156863,-0.045098037,0.1156863,-0.021568626,-0.16666666,0.18627453,0.041176498,-0.08823529,0.2529412,0.100000024,-0.029411763,0.15098041,0.017647088,-0.119607836,0.08431375,-0.041176468,-0.1745098,0.17843139,0.03333336,-0.10392156,0.2764706,0.127451,-0.0372549,0.35490197,0.2019608,0.001960814,0.14705884,-0.0058823526,-0.1862745,0.22549021,0.05686277,-0.11568627,0.4254902,0.22156864,0.03725493,0.43333334,0.22941178,0.041176498,0.40588236,0.20980394,0.0058823824,0.45686275,0.24901962,0.013725519,0.45686275,0.23333335,-0.009803921,0.327451,0.096078455,-0.123529404,0.24117649,0.013725519,-0.16274509,0.15490198,-0.045098037,-0.20196077,0.21764708,0.03725493,-0.123529404,0.25686276,0.08823532,-0.072549015,0.32352942,0.16274512,0.001960814,0.31176472,0.123529434,-0.03333333,0.49607843,0.26078433,0.08823532,0.3509804,0.15098041,-0.0372549,0.23725492,0.049019635,-0.13921568,0.24509805,0.05686277,-0.13137254,0.3,0.100000024,-0.096078426,0.35490197,0.14705884,-0.06078431,0.2764706,0.08431375,-0.11568627,0.38235295,0.19803923,0.001960814,0.37058824,0.19803923,0.009803951,0.20980394,0.072549045,-0.08431372,0.15490198,0.041176498,-0.072549015,0.23333335,0.10784316,-0.0058823526,0.127451,-0.0058823526,-0.13529411,0.08039218,-0.05686274,-0.19411764,0.049019635,-0.08039215,-0.20196077,0.1901961,0.03333336,-0.11568627,0.22156864,0.06862748,-0.10392156,0.17843139,0.045098066,-0.13529411,0.13137257,-0.013725489,-0.20196077,0.21764708,0.06078434,-0.1117647,0.46862745,0.2764706,0.09215689,0.46862745,0.27254903,0.09215689,0.40588236,0.2137255,0.021568656,0.36666667,0.18235296,-0.021568626,0.45294118,0.22156864,-0.021568626,0.22941178,0.009803951,-0.19411764,0.19803923,-0.017647058,-0.1862745,0.15882355,-0.029411763,-0.17843136,0.14705884,-0.009803921,-0.15490195,0.16274512,0.029411793,-0.119607836,0.25686276,0.119607866,-0.03333333,0.30784315,0.123529434,-0.03333333,0.48039216,0.2529412,0.07647061,0.3627451,0.15882355,-0.021568626,0.38235295,0.17843139,-0.017647058,0.37843138,0.17450982,-0.025490195,0.33137256,0.13137257,-0.06862745,0.3156863,0.1156863,-0.08823529,0.29607844,0.10784316,-0.08823529,0.2764706,0.10784316,-0.08039215,0.44117647,0.2647059,0.06470591,0.34705883,0.19411767,0.017647088,0.4490196,0.3,0.13529414,0.23725492,0.100000024,-0.0372549,0.100000024,-0.0372549,-0.17843136,0.08823532,-0.049019605,-0.18235293,0.08823532,-0.045098037,-0.16274509,0.19803923,0.03333336,-0.123529404,0.23725492,0.06862748,-0.10784313,0.3156863,0.15490198,-0.049019605,0.2137255,0.06470591,-0.13529411,0.23333335,0.08431375,-0.09215686,0.42941177,0.2529412,0.07647061,0.44509804,0.2647059,0.08823532,0.41764706,0.24117649,0.052941203,0.42941177,0.24509805,0.025490224,0.39803922,0.17450982,-0.06862745,0.22156864,0.009803951,-0.1862745,0.15490198,-0.0372549,-0.1980392,0.123529434,-0.041176468,-0.18235293,0.08431375,-0.049019605,-0.18235293,0.119607866,0.001960814,-0.13921568,0.22549021,0.096078455,-0.052941173,0.32352942,0.15882355,0.001960814,0.3745098,0.17450982,0.001960814,0.38627452,0.18627453,0.0058823824,0.38627452,0.1901961,0.001960814,0.29607844,0.10784316,-0.072549015,0.19411767,0.017647088,-0.15490195,0.2647059,0.07647061,-0.1117647,0.38627452,0.19411767,-0.0058823526,0.4019608,0.22156864,0.03333336,0.39019608,0.22549021,0.041176498,0.30784315,0.16274512,-0.009803921,0.3156863,0.18627453,0.017647088,0.2019608,0.072549045,-0.06862745,0.1156863,-0.013725489,-0.15098038,0.13921571,-0.0019607842,-0.13921568,0.2137255,0.06078434,-0.072549015,0.26078433,0.100000024,-0.08431372,0.32352942,0.15490198,-0.041176468,0.4372549,0.26078433,0.041176498,0.3627451,0.19411767,-0.013725489,0.27254903,0.1156863,-0.07647058,0.3509804,0.19411767,0.013725519,0.39411765,0.23333335,0.06078434,0.38235295,0.2137255,0.03333336,0.37843138,0.19411767,-0.013725489,0.38627452,0.16666669,-0.072549015,0.28431374,0.06862748,-0.123529404,0.15098041,-0.03333333,-0.17843136,0.16666669,0.009803951,-0.13137254,0.08431375,-0.029411763,-0.16666666,0.13921571,0.029411793,-0.1117647,0.18627453,0.072549045,-0.06862745,0.39803922,0.23333335,0.05686277,0.3,0.119607866,-0.05686274,0.40980393,0.2137255,0.029411793,0.34705883,0.15882355,-0.021568626,0.24901962,0.07647061,-0.09215686,0.3,0.10784316,-0.072549015,0.15490198,-0.009803921,-0.1745098,0.100000024,-0.049019605,-0.20588234,0.2882353,0.123529434,-0.041176468,0.39411765,0.22549021,0.05686277,0.2137255,0.08039218,-0.072549015,0.15882355,0.041176498,-0.11568627,0.2647059,0.13529414,-0.013725489,0.17058825,0.045098066,-0.09215686,0.17843139,0.025490224,-0.11568627,0.14313728,-0.013725489,-0.15098038,0.3,0.14313728,-0.052941173,0.31176472,0.14705884,-0.052941173,0.38627452,0.20980394,-0.0019607842,0.43333334,0.24509805,0.025490224,0.3,0.13529414,-0.06862745,0.17450982,0.029411793,-0.14313725,0.28039217,0.123529434,-0.05686274,0.38235295,0.2137255,0.025490224,0.40980393,0.2019608,-0.017647058,0.35490197,0.127451,-0.099999994,0.31176472,0.09215689,-0.10392156,0.16666669,-0.0058823526,-0.16666666,0.22941178,0.07647061,-0.052941173,0.10392159,0.0058823824,-0.10392156,0.13137257,0.03725493,-0.08039215,0.20588237,0.08823532,-0.049019605,0.2647059,0.1156863,-0.045098037,0.3156863,0.13921571,-0.0372549,0.28039217,0.1156863,-0.052941173,0.15882355,-0.009803921,-0.17058823,0.23333335,0.06078434,-0.10392156,0.29607844,0.10392159,-0.07647058,0.24117649,0.06862748,-0.09215686,0.2137255,0.052941203,-0.096078426,0.21764708,0.06078434,-0.096078426,0.23333335,0.09215689,-0.05686274,0.127451,0.0058823824,-0.13529411,0.16666669,0.049019635,-0.08431372,0.18235296,0.06078434,-0.05686274,0.31960785,0.17058825,0.03725493,0.24901962,0.09215689,-0.052941173,0.27254903,0.11176473,-0.03333333,0.24901962,0.100000024,-0.08823529,0.26862746,0.11176473,-0.08823529,0.30784315,0.13529414,-0.06470588,0.3392157,0.15882355,-0.049019605,0.39803922,0.20980394,-0.0058823526,0.327451,0.15882355,-0.0372549,0.31960785,0.15882355,-0.03333333,0.38627452,0.2137255,0.017647088,0.4019608,0.18627453,-0.0372549,0.33137256,0.10392159,-0.119607836,0.24117649,0.041176498,-0.14705881,0.15490198,0.001960814,-0.14705881,0.14705884,0.017647088,-0.099999994,0.17843139,0.05686277,-0.05686274,0.20588237,0.08431375,-0.041176468,0.27254903,0.13529414,-0.0058823526,0.39411765,0.22941178,0.06862748,0.38235295,0.2137255,0.03333336,0.3509804,0.1901961,0.001960814,0.25686276,0.096078455,-0.08431372,0.23725492,0.06470591,-0.119607836,0.28039217,0.08823532,-0.09215686,0.33137256,0.14313728,-0.045098037,0.34705883,0.16666669,-0.021568626,0.33137256,0.16666669,0.013725519,0.05686277,-0.05686274,-0.17843136,0.13137257,0.017647088,-0.119607836,0.072549045,-0.0372549,-0.14313725,0.22156864,0.08823532,-0.029411763,0.36666667,0.2019608,0.06470591,0.3,0.13921571,0.001960814,0.25686276,0.07647061,-0.0372549,0.19803923,0.049019635,-0.13137254,0.16666669,0.017647088,-0.15882352,0.26078433,0.096078455,-0.09215686,0.22941178,0.06078434,-0.13137254,0.2647059,0.08823532,-0.11568627,0.3509804,0.17450982,-0.03333333,0.30784315,0.15490198,-0.03333333,0.3,0.14705884,-0.03333333,0.33137256,0.13921571,-0.06470588,0.2647059,0.06078434,-0.15098038,0.20980394,0.017647088,-0.17843136,0.20588237,0.03333336,-0.13921568,0.21764708,0.06862748,-0.10392156,0.07647061,-0.05686274,-0.20980391,0.11176473,-0.013725489,-0.16274509,0.26862746,0.13137257,-0.03333333,0.24117649,0.096078455,-0.06078431,0.2647059,0.13921571,-0.029411763,0.30392158,0.15098041,-0.025490195,0.3156863,0.15098041,-0.041176468,0.22549021,0.05686277,-0.13921568,0.2529412,0.07647061,-0.119607836,0.4137255,0.21764708,0.013725519,0.39803922,0.20588237,0.013725519,0.30392158,0.14313728,-0.009803921,0.23333335,0.08823532,-0.041176468,0.26862746,0.127451,-0.017647058,0.2647059,0.123529434,-0.029411763,0.20980394,0.07647061,-0.072549015,0.30784315,0.15882355,0.001960814,0.23725492,0.08039218,-0.06078431,0.30784315,0.13529414,0.025490224,0.21764708,0.06078434,-0.1117647,0.23333335,0.06862748,-0.11568627,0.3392157,0.15882355,-0.041176468,0.2647059,0.08431375,-0.11568627,0.23333335,0.05686277,-0.14705881,0.3509804,0.17450982,-0.0372549,0.43333334,0.2647059,0.045098066,0.36666667,0.20588237,0.0058823824,0.32352942,0.15882355,-0.025490195,0.27254903,0.11176473,-0.072549015,0.127451,-0.029411763,-0.19411764,0.23333335,0.03725493,-0.14313725,0.22549021,0.025490224,-0.14705881,0.26862746,0.06470591,-0.1117647,0.32352942,0.1156863,-0.06078431,0.3,0.1156863,-0.052941173,0.23725492,0.08039218,-0.072549015,0.22549021,0.08823532,-0.06470588,0.19411767,0.05686277,-0.096078426,0.30392158,0.13921571,-0.03333333,0.40980393,0.22941178,0.03333336,0.40588236,0.22156864,0.029411793,0.4882353,0.28431374,0.096078455,0.37058824,0.18235296,0.021568656,0.3392157,0.16666669,0.017647088,0.33137256,0.17058825,0.029411793,0.20588237,0.06470591,-0.08431372,0.22156864,0.08823532,-0.08431372,0.20980394,0.072549045,-0.08431372,0.27254903,0.119607866,-0.029411763,0.2019608,0.045098066,-0.07647058,0.2137255,0.06078434,-0.05686274,0.16274512,0.017647088,-0.15882352,0.24117649,0.072549045,-0.11568627,0.3392157,0.14705884,-0.06078431,0.2764706,0.09215689,-0.11568627,0.24901962,0.06862748,-0.13529411,0.2529412,0.07647061,-0.12745097,0.35882354,0.17843139,-0.029411763,0.48039216,0.31176472,0.08823532,0.5,0.33529413,0.123529434,0.4764706,0.3,0.100000024,0.28431374,0.10784316,-0.07647058,0.30392158,0.08039218,-0.123529404,0.29215688,0.08039218,-0.12745097,0.2764706,0.045098066,-0.15490195,0.22549021,0.0058823824,-0.1862745,0.24901962,0.049019635,-0.12745097,0.31960785,0.13921571,-0.029411763,0.21764708,0.072549045,-0.07647058,0.123529434,-0.0058823526,-0.14313725,0.15490198,0.021568656,-0.11568627,0.25686276,0.10392159,-0.052941173,0.28039217,0.123529434,-0.041176468,0.35882354,0.18627453,0.03725493,0.30784315,0.13921571,0.0058823824,0.2647059,0.10784316,-0.017647058,0.23333335,0.09215689,-0.06470588,0.14313728,0.021568656,-0.15882352,0.096078455,-0.025490195,-0.17843136,0.18235296,0.041176498,-0.099999994,0.27254903,0.10784316,-0.03333333,0.19803923,0.03725493,-0.08823529,0.127451,-0.017647058,-0.13137254,0.13529414,-0.0058823526,-0.17843136,0.20588237,0.049019635,-0.14313725,0.2764706,0.10392159,-0.10392156,0.29215688,0.10784316,-0.099999994,0.3,0.1156863,-0.09215686,0.3,0.1156863,-0.08431372,0.38235295,0.2019608,-0.009803921,0.30784315,0.14705884,-0.049019605,0.4254902,0.24901962,0.049019635,0.46470588,0.28431374,0.08039218,0.26078433,0.09215689,-0.08823529,0.029411793,-0.11568627,-0.25686276,0.30392158,0.08431375,-0.11568627,0.4137255,0.15882355,-0.072549015,0.24901962,0.029411793,-0.16666666,0.28039217,0.07647061,-0.10392156,0.33529413,0.14705884,-0.025490195,0.22941178,0.072549045,-0.08431372,0.18235296,0.045098066,-0.08823529,0.31960785,0.18235296,0.041176498,0.21764708,0.09215689,-0.03333333,0.17058825,0.049019635,-0.072549015,0.3745098,0.21764708,0.072549045,0.2764706,0.13529414,0.001960814,0.16666669,0.03333336,-0.09215686,0.20980394,0.072549045,-0.072549015,0.08039218,-0.03333333,-0.20980391,0.13921571,0.0058823824,-0.14313725,0.22549021,0.06862748,-0.07647058,0.33529413,0.15882355,-0.013725489,0.15882355,-0.0058823526,-0.12745097,0.1156863,-0.03333333,-0.15098038,0.23725492,0.08431375,-0.1117647,0.18627453,0.03333336,-0.16274509,0.23333335,0.072549045,-0.13529411,0.38235295,0.18235296,-0.041176468,0.30392158,0.11176473,-0.099999994,0.2882353,0.11176473,-0.08431372,0.21764708,0.049019635,-0.13529411,0.39019608,0.20980394,0.017647088,0.4490196,0.25686276,0.05686277,0.42941177,0.23333335,0.029411793,0.44509804,0.23333335,0.017647088,0.4019608,0.18627453,-0.03333333,0.2647059,0.052941203,-0.15490195,0.2529412,0.029411793,-0.1862745,0.28039217,0.06078434,-0.14313725,0.31176472,0.11176473,-0.072549015,0.29607844,0.123529434,-0.041176468,0.10392159,-0.029411763,-0.18235293,0.06862748,-0.05686274,-0.20588234,0.3627451,0.2019608,0.021568656,0.3509804,0.19803923,0.017647088,0.2529412,0.1156863,-0.049019605,0.31176472,0.16666669,0.001960814,0.3392157,0.19411767,0.03333336,0.2764706,0.13137257,-0.025490195,0.20588237,0.06862748,-0.08431372,0.10784316,-0.017647058,-0.1862745,0.1901961,0.045098066,-0.119607836,0.29215688,0.127451,-0.05686274,0.23333335,0.06078434,-0.11568627,0.123529434,-0.0372549,-0.1745098,0.2019608,0.03333336,-0.1117647,0.39803922,0.22156864,-0.029411763,0.33529413,0.15882355,-0.08431372,0.25686276,0.08039218,-0.14313725,0.39019608,0.18627453,-0.049019605,0.2137255,0.029411793,-0.17058823,0.29215688,0.100000024,-0.09215686,0.28039217,0.10784316,-0.06078431,0.48431373,0.2882353,0.096078455,0.45294118,0.24117649,0.03725493,0.40588236,0.19803923,-0.009803921,0.40980393,0.20588237,-0.009803921,0.49607843,0.30392158,0.06862748,0.4372549,0.22941178,0.009803951,0.25686276,0.07647061,-0.11568627,0.29215688,0.100000024,-0.08823529,0.38235295,0.17450982,-0.017647058,0.5,0.31960785,0.11176473,0.2764706,0.11176473,-0.06862745,0.021568656,-0.09215686,-0.22549018,0.1901961,0.052941203,-0.1117647,0.327451,0.16666669,-0.009803921,0.30392158,0.15490198,-0.013725489,0.29607844,0.15882355,0.001960814,0.4019608,0.2529412,0.08039218,0.21764708,0.08823532,-0.072549015,0.18627453,0.052941203,-0.1117647,0.17058825,0.029411793,-0.13529411,0.30392158,0.13921571,-0.06470588,0.22549021,0.06862748,-0.12745097,0.20588237,0.03333336,-0.18235293,0.21764708,0.045098066,-0.15882352,0.25686276,0.09215689,-0.10392156,0.28039217,0.1156863,-0.119607836,0.3745098,0.19411767,-0.05686274,0.35490197,0.17058825,-0.06470588,0.2882353,0.10784316,-0.10784313,0.025490224,-0.1117647,-0.2647059,0.17058825,0.0058823824,-0.15882352,0.29607844,0.1156863,-0.049019605,0.49215686,0.28431374,0.096078455,0.49215686,0.26862746,0.049019635,0.49215686,0.26078433,0.03725493,0.48431373,0.26078433,0.03725493,0.49215686,0.29607844,0.072549045,0.5,0.33529413,0.10784316,0.45686275,0.2764706,0.06862748,0.41764706,0.22941178,0.03333336,0.49607843,0.28431374,0.08039218,0.40588236,0.18627453,-0.009803921,0.327451,0.13529414,-0.06470588,0.43333334,0.23725492,0.03725493,0.5,0.3,0.10392159,0.37058824,0.19803923,0.03333336,0.041176498,-0.08431372,-0.22156861,0.35882354,0.20588237,0.05686277,0.28039217,0.13529414,-0.013725489,0.2647059,0.123529434,-0.041176468,0.22549021,0.08039218,-0.08431372,0.20588237,0.06078434,-0.09215686,0.28039217,0.123529434,-0.08039215,0.1901961,0.029411793,-0.1980392,0.2137255,0.049019635,-0.18235293,0.28431374,0.10784316,-0.10392156,0.3627451,0.17843139,-0.03333333,0.14313728,-0.0058823526,-0.20196077,0.20980394,0.052941203,-0.15490195,0.3392157,0.15882355,-0.06862745,0.3627451,0.17450982,-0.049019605,0.31960785,0.13137257,-0.07647058,0.26078433,0.08431375,-0.09215686,0.327451,0.13921571,-0.0372549,0.37058824,0.17450982,-0.009803921,0.4764706,0.26862746,0.06470591,0.48039216,0.2764706,0.06470591,0.4254902,0.22156864,0.009803951,0.47254902,0.2764706,0.05686277,0.44509804,0.2647059,0.041176498,0.5,0.3509804,0.123529434,0.44509804,0.24901962,0.045098066,0.48039216,0.26078433,0.05686277,0.42941177,0.20588237,0.009803951,0.21764708,0.029411793,-0.13921568,0.35882354,0.14705884,-0.03333333,0.45294118,0.22941178,0.049019635,0.42941177,0.22941178,0.072549045,0.33529413,0.16666669,0.03333336,0.3392157,0.18627453,0.045098066,0.2882353,0.13921571,0.001960814,0.28431374,0.13137257,-0.021568626,0.3392157,0.17843139,0.009803951,0.2764706,0.127451,-0.017647058,0.29215688,0.13921571,-0.0372549,0.17843139,0.025490224,-0.17058823,0.19411767,0.045098066,-0.15098038,0.23725492,0.08039218,-0.10784313,0.33529413,0.16274512,-0.017647058,0.24901962,0.06862748,-0.1117647,0.19803923,0.021568656,-0.15490195,0.2882353,0.10392159,-0.08431372,0.34705883,0.15490198,-0.041176468,0.30784315,0.119607866,-0.07647058,0.35882354,0.16274512,-0.029411763,0.39803922,0.19803923,0.001960814,0.3,0.119607866,-0.05686274,0.47254902,0.27254903,0.072549045,0.48039216,0.2764706,0.06862748,0.43333334,0.23725492,0.025490224,0.4490196,0.25686276,0.03725493,0.31960785,0.14705884,-0.06862745,0.45294118,0.24901962,0.021568656,0.42941177,0.20980394,-0.009803921,0.35490197,0.14313728,-0.049019605,0.40588236,0.16666669,-0.03333333,0.3156863,0.07647061,-0.123529404,0.39019608,0.14705884,-0.06470588,0.38627452,0.15098041,-0.052941173,0.4882353,0.2529412,0.045098066,0.31176472,0.119607866,-0.03333333,0.20980394,0.06078434,-0.07647058,0.30784315,0.15098041,0.017647088,0.28039217,0.127451,-0.009803921,0.34313726,0.18235296,0.017647088,0.39803922,0.24117649,0.07647061,0.33137256,0.17058825,0.0058823824,0.30784315,0.14705884,-0.029411763,0.24901962,0.09215689,-0.08039215,0.2882353,0.123529434,-0.05686274,0.30392158,0.13529414,-0.03333333,0.27254903,0.072549045,-0.07647058,0.2764706,0.10392159,-0.05686274,0.2882353,0.10392159,-0.05686274,0.24509805,0.05686277,-0.1117647,0.17450982,0.0058823824,-0.15490195,0.2882353,0.100000024,-0.06862745,0.39803922,0.19411767,0.013725519,0.41764706,0.22156864,0.03725493,0.43333334,0.22549021,0.029411793,0.3392157,0.14313728,-0.05686274,0.44509804,0.23333335,0.021568656,0.49607843,0.30392158,0.07647061,0.3392157,0.16274512,-0.045098037,0.327451,0.13137257,-0.07647058,0.4254902,0.20588237,-0.009803921,0.4372549,0.2019608,0.001960814,0.39803922,0.15882355,-0.041176468,0.33137256,0.08823532,-0.11568627,0.43333334,0.18627453,-0.0372549,0.4882353,0.24509805,0.013725519,0.49215686,0.26078433,0.03725493,0.28431374,0.100000024,-0.06470588,0.16274512,0.021568656,-0.119607836,0.18627453,0.045098066,-0.07647058,0.39803922,0.23725492,0.096078455,0.28431374,0.14313728,-0.0058823526,0.20588237,0.072549045,-0.123529404,0.22549021,0.08431375,-0.096078426,0.31960785,0.15882355,0.001960814,0.3156863,0.15490198,0.0058823824,0.26078433,0.10784316,-0.052941173,0.20980394,0.06470591,-0.08823529,0.15490198,-0.03333333,-0.15882352,0.23333335,0.041176498,-0.10784313,0.28039217,0.07647061,-0.08039215,0.22549021,0.03333336,-0.123529404,0.22549021,0.049019635,-0.10784313,0.34705883,0.15098041,-0.017647058,0.4372549,0.22156864,0.041176498,0.5,0.29607844,0.10392159,0.35490197,0.123529434,-0.072549015,0.37058824,0.13529414,-0.07647058,0.42156863,0.18235296,-0.0372549,0.37058824,0.14313728,-0.07647058,0.29607844,0.08431375,-0.123529404,0.47254902,0.24117649,0.021568656,0.31176472,0.100000024,-0.08823529,0.30392158,0.09215689,-0.08431372,0.30784315,0.09215689,-0.08823529,0.40980393,0.17450982,-0.021568626,0.40980393,0.17450982,-0.03333333,0.37843138,0.15882355,-0.05686274,0.37058824,0.16274512,-0.03333333,0.28431374,0.11176473,-0.052941173,0.1901961,0.06078434,-0.099999994,0.09215689,-0.029411763,-0.15490195,0.3392157,0.19411767,0.06078434,0.15098041,0.03333336,-0.10392156,0.05686277,-0.06862745,-0.22549018,0.13529414,-0.0019607842,-0.14705881,0.14705884,0.009803951,-0.123529404,0.2647059,0.1156863,-0.017647058,0.3,0.14705884,0.009803951,0.23725492,0.10392159,-0.029411763,0.37058824,0.15098041,-0.025490195,0.23725492,0.049019635,-0.11568627,0.21764708,0.025490224,-0.13529411,0.36666667,0.15490198,-0.029411763,0.39411765,0.17843139,-0.009803921,0.44509804,0.22549021,0.029411793,0.46470588,0.24117649,0.041176498,0.24901962,0.06470591,-0.10392156,0.20980394,-0.009803921,-0.20196077,0.14313728,-0.08039215,-0.28039217,0.31176472,0.08039218,-0.123529404,0.31176472,0.08039218,-0.123529404,0.3,0.07647061,-0.123529404,0.30784315,0.096078455,-0.096078426,0.26078433,0.06078434,-0.1117647,0.2137255,0.025490224,-0.13137254,0.16274512,-0.009803921,-0.16274509,0.3392157,0.13529414,-0.041176468,0.31176472,0.10784316,-0.08039215,0.25686276,0.072549045,-0.119607836,0.21764708,0.041176498,-0.123529404,0.15098041,-0.0019607842,-0.14705881,0.123529434,-0.009803921,-0.15490195,0.1156863,-0.009803921,-0.13137254,0.41764706,0.26862746,0.127451,0.17843139,0.05686277,-0.08431372,0.08431375,-0.041176468,-0.19411764,0.08431375,-0.049019605,-0.19019607,0.11176473,-0.017647058,-0.14705881,0.3156863,0.15882355,0.025490224,0.30784315,0.16274512,0.03725493,0.31176472,0.17058825,0.045098066,}\
;

int model_test(void)
{
	mat_t x = {
		.dims = { 1, 768 },
#ifdef USE_VECTORIZATION
		.row_major = 1,
#endif
	};

	nn_layer_t L[] = {
		{
			.w = nn_mat_load_row_order("data/model_fc0/dense.kernel", 0),
			.b = nn_mat_load_row_order("data/model_fc0/dense.bias", 1),
			.activation = nn_act_relu
		},
		{
			.w = nn_mat_load_row_order("data/model_fc0/dense_1.kernel", 0),
			.b = nn_mat_load_row_order("data/model_fc0/dense_1.bias", 1),
			.activation = nn_act_softmax
		},
		{}
	};

	// Allocate and setup layers and matrices
	assert(nn_mat_init(&x) == 0);
	assert(nn_init(L, &x) == 0);

for (int i = 768; i--;) *nn_mat_e(&x, 0, i) = hay[i];

	mat_t* y;
	time_t start = time(NULL);
	while(time(NULL) == start);
	start = time(NULL);

	int cycles = 0;
	while(time(NULL) == start)
	{
		y = nn_predict(L, &x);
		++cycles;
	}

    int passed = y->data.f[0] < y->data.f[1] && y->data.f[2] < y->data.f[1];

	Log("%d cps", 1, cycles);
	Log("%f %f %f", passed,
	y->data.f[0],
	y->data.f[1],
	y->data.f[2]);

	return !passed;
}

TEST_BEGIN
	.name = "fc_model_test",
	.description = "Loads a one layer fully connected NN and produces a hypothesis on test input",
	.run = model_test,
TEST_END
