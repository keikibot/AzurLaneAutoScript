from module.base.button import ButtonGrid
from module.base.decorator import Config
from module.base.utils import *
from module.equipment.assets import *
from module.equipment.equipment import Equipment
from module.logger import logger
from module.ui.scroll import Scroll

EQUIP_INFO_BAR = ButtonGrid(
    origin=(723, 111), delta=(94, 0), button_shape=(76, 76), grid_shape=(5, 1), name="EQUIP_INFO_BAR"
)

EQUIPMENT_SCROLL = Scroll(EQUIP_SCROLL, color=(
    247, 211, 66), name='EQUIP_SCROLL')

SIM_VALUE = 0.90


class EquipmentChange(Equipment):

    equip_list = {}
    equipping_list = [0, 1, 2, 3, 4]

    def get_equiping_list(self, skip_first_screenshot=True):
        '''
        Pages:
            in: ship's equipments details
        '''
        logger.info("Get equipping list")
        if skip_first_screenshot:
            pass
        else:
            self.device.screenshot()
        for index in range(0, 5):
            enter_button = globals()[
                'EQUIP_TAKE_ON_{index}'.format(index=index)]
            if self.appear(enter_button, offset=(5, 5)):
                self.equipping_list.remove(index)
        logger.info(f"Equipping list: {self.equipping_list}")

    def record_equipment(self, index_list=range(0, 5), skip_first_screenshot=True):
        '''
        Record equipment through upgrade page
        Notice: The equipment icons in the upgrade page are the same size as the icons in the equipment status
        '''
        logger.info('RECORD EQUIPMENT')
        self.equip_side_navbar_ensure(bottom=2)
        self.get_equiping_list()
        self.equip_side_navbar_ensure(bottom=1)

        for index in index_list:
            if index in self.equipping_list:
                logger.info(f'Record {index}')
                while 1:
                    if skip_first_screenshot:
                        skip_first_screenshot = False
                    else:
                        self.device.screenshot()

                    if self.appear(EQUIPMENT_OPEN, interval=3):
                        self.device.click(EQUIP_INFO_BAR[(index, 0)])
                        continue
                    if self.appear_then_click(UPGRADE_ENTER, interval=3):
                        continue
                    if self.appear(UPGRADE_ENTER_CHECK, interval=3):
                        self.equip_list[index] = self.image_area(EQUIP_SAVE)
                        self.ui_click(
                            click_button=UPGRADE_QUIT, check_button=EQUIPMENT_OPEN, appear_button=UPGRADE_ENTER_CHECK, skip_first_screenshot=True)
                        break

    def equipment_take_on(self, index_list=range(0, 5), skip_first_screenshot=True):
        '''
        Equip the equipment previously recorded
        '''
        logger.info('Take on equipment')
        self.equip_side_navbar_ensure(bottom=2)

        self.ensure_no_info_bar(1)

        for index in index_list:
            if index in self.equipping_list:
                logger.info(f'Take on {index}')
                enter_button = globals()[
                    'EQUIP_TAKE_ON_{index}'.format(index=index)]

                self.ui_click(enter_button, check_button=EQUIPPING_ON,
                              skip_first_screenshot=skip_first_screenshot, offset=(5, 5))
                self._find_equip(index)

        self.equipping_list = [0, 1, 2, 3, 4]

    @Config.when(DEVICE_CONTROL_METHOD='minitouch')
    def _equipment_swipe(self, distance=190):
        # Distance of two commission is 146px
        p1, p2 = random_rectangle_vector(
            (0, -distance), box=(620, 67, 1154, 692), random_range=(-20, -5, 20, 5))
        self.device.drag(p1, p2, segments=2, shake=(25, 0),
                         point_random=(0, 0, 0, 0), shake_random=(-5, 0, 5, 0))
        self.device.sleep(0.3)
        self.device.screenshot()

    @Config.when(DEVICE_CONTROL_METHOD=None)
    def _equipment_swipe(self, distance=300):
        # Distance of two commission is 146px
        p1, p2 = random_rectangle_vector(
            (0, -distance), box=(620, 67, 1154, 692), random_range=(-20, -5, 20, 5))
        self.device.drag(p1, p2, segments=2, shake=(25, 0),
                         point_random=(0, 0, 0, 0), shake_random=(-5, 0, 5, 0))
        self.device.sleep(0.3)
        self.device.screenshot()

    def _equip_equipment(self, point, offset=(100, 100), skip_first_screenshot=True):
        '''
        Equip Equipment then back to ship details
        Confirm the popup
        Pages:
            in: EQUIPMENT STATUS
            out: SHIP_SIDEBAR_EQUIPMENT
        '''

        have_equipped = False

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if not have_equipped and self.appear(EQUIPPING_OFF, interval=5):
                self.device.click(
                    Button(button=(point[0], point[1], point[0]+offset[0], point[1]+offset[1]), color=None, area=None))
                have_equipped = True
                continue
            if have_equipped and self.appear_then_click(EQUIP_CONFIRM, interval=2):
                continue
            if self.info_bar_count():
                break

    def _find_equip(self, index):
        '''
        Find the equipment previously recorded 
        Pages:
            in: EQUIPMENT STATUS
        '''

        self.equipping_set(False)

        res = cv2.matchTemplate(np.array(self.device.screenshot()), np.array(
            self.equip_list[index]), cv2.TM_CCOEFF_NORMED)
        _, sim, _, point = cv2.minMaxLoc(res)

        if sim > SIM_VALUE:
            self._equip_equipment(point)
            return

        for _ in range(0, 15):
            self._equipment_swipe()

            res = cv2.matchTemplate(np.array(self.device.screenshot()), np.array(
                self.equip_list[index]), cv2.TM_CCOEFF_NORMED)
            _, sim, _, point = cv2.minMaxLoc(res)

            if sim > SIM_VALUE:
                self._equip_equipment(point)
                break
            if self.appear(EQUIPMENT_SCROLL_BOTTOM):
                logger.warning('No recorded equipment was found.')
                self.ui_back(check_button=globals()[
                             f'EQUIP_TAKE_ON_{index}'], appear_button=EQUIPPING_OFF)
                break

        return
