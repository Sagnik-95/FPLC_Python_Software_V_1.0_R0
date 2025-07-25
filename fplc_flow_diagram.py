from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsDropShadowEffect, QFileDialog
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QLinearGradient, QImage
import sys
import math

class FPLCSystemView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Professional FPLC color scheme
        self.colors = {
            'buffer': QColor('#2980b9'),
            'sample': QColor('#27ae60'),
            'pump': QColor('#8e44ad'),
            'column': QColor('#c0392b'),
            'detector': QColor('#f39c12'),
            'collector': QColor('#16a085'),
            'valve': QColor('#2c3e50'),
            'waste': QColor('#7f8c8d')
        }
        
        # Layout configuration
        self.center_x, self.center_y = 600, 400
        self.radius = 250
        self.box_width = 180
        self.box_height = 90
        
        self.setup_components()
        self.setup_flow_paths()
        self.draw_fplc_diagram()

    def setup_components(self):
        self.components = {
            'buffer_a': {
                'text': 'Buffer A\nReservoir',
                'type': 'buffer',
                'pos': (200, 100)
            },
            'buffer_b': {
                'text': 'Buffer B\nReservoir',
                'type': 'buffer',
                'pos': (400, 100)
            },
            'sample': {
                'text': 'Sample\nInjection',
                'type': 'sample',
                'pos': (600, 100)
            },
            'pump_a': {
                'text': 'Pump A',
                'type': 'pump',
                'pos': (200, 250)
            },
            'pump_b': {
                'text': 'Pump B',
                'type': 'pump',
                'pos': (400, 250)
            },
            'mixer': {
                'text': 'Mixer',
                'type': 'valve',
                'pos': (300, 400)
            },
            'injection_valve': {
                'text': 'Injection\nValve',
                'type': 'valve',
                'pos': (500, 400)
            },
            'column': {
                'text': 'Chromatography\nColumn',
                'type': 'column',
                'pos': (700, 400)
            },
            'uv_detector': {
                'text': 'UV\nDetector',
                'type': 'detector',
                'pos': (900, 400)
            },
            'conductivity': {
                'text': 'Conductivity\nMonitor',
                'type': 'detector',
                'pos': (900, 250)
            },
            'fraction_collector': {
                'text': 'Fraction\nCollector',
                'type': 'collector',
                'pos': (1000, 550)
            },
            'waste': {
                'text': 'Waste',
                'type': 'waste',
                'pos': (800, 550)
            }
        }

    def setup_flow_paths(self):
        self.flow_paths = [
            ('buffer_a', 'pump_a'),
            ('buffer_b', 'pump_b'),
            ('pump_a', 'mixer'),
            ('pump_b', 'mixer'),
            ('mixer', 'injection_valve'),
            ('sample', 'injection_valve'),
            ('injection_valve', 'column'),
            ('column', 'uv_detector'),
            ('uv_detector', 'conductivity'),
            ('conductivity', 'fraction_collector'),
            ('conductivity', 'waste')
        ]

    def draw_3d_component(self, x, y, text, color):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(5, 5)
        shadow.setColor(QColor(0, 0, 0, 80))
        
        gradient = QLinearGradient(x - self.box_width/2, y - self.box_height/2,
                                 x + self.box_width/2, y + self.box_height/2)
        gradient.setColorAt(0, color.lighter(130))
        gradient.setColorAt(1, color.darker(110))
        
        path = QPainterPath()
        path.addRoundedRect(x - self.box_width/2, y - self.box_height/2,
                          self.box_width, self.box_height, 15, 15)
        
        box = self.scene.addPath(path, QPen(Qt.GlobalColor.black, 2), QBrush(gradient))
        box.setGraphicsEffect(shadow)
        
        # Multi-line text support
        lines = text.split('\n')
        y_offset = -(len(lines) - 1) * 12
        
        for line in lines:
            text_item = self.scene.addText(line, QFont("Arial", 12, QFont.Weight.Bold))
            text_item.setDefaultTextColor(Qt.GlobalColor.white)
            text_width = text_item.boundingRect().width()
            text_height = text_item.boundingRect().height()
            text_item.setPos(x - text_width/2, y + y_offset - text_height/2)
            y_offset += 24

    def draw_flow_line(self, start_pos, end_pos, color=QColor('#2c3e50')):
        # Calculate box dimensions
        box_w = self.box_width / 2
        box_h = self.box_height / 2
    
        # Calculate start and end points adjusted for box boundaries
        start_x, start_y = start_pos
        end_x, end_y = end_pos
    
        # Calculate direction vectors
        dx = end_x - start_x
        dy = end_y - start_y
    
        # Normalize direction vector
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx /= length
            dy /= length
    
        # Adjust start and end points to box boundaries
        start_point = QPointF(
            start_x + dx * box_w,
            start_y + dy * box_h
        )
        end_point = QPointF(
            end_x - dx * box_w,
            end_y - dy * box_h
        )
    
        # Calculate control points for curved flow
        ctrl1 = QPointF(
            start_point.x() + (end_point.x() - start_point.x()) / 3,
            start_point.y() + (end_point.y() - start_point.y()) / 3
        )
        ctrl2 = QPointF(
            start_point.x() + 2 * (end_point.x() - start_point.x()) / 3,
            start_point.y() + 2 * (end_point.y() - start_point.y()) / 3
        )
    
        # Draw the curved path
        path = QPainterPath()
        path.moveTo(start_point)
        path.cubicTo(ctrl1, ctrl2, end_point)
    
        # Flow line with arrow
        pen = QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        self.scene.addPath(path, pen)
    
        # Add arrow head
        angle = math.atan2(end_point.y() - ctrl2.y(), end_point.x() - ctrl2.x())
        arrow_size = 10
        arrow_p1 = QPointF(
            end_point.x() - arrow_size * math.cos(angle - math.pi/5),
            end_point.y() - arrow_size * math.sin(angle - math.pi/6)
        )
        arrow_p2 = QPointF(
            end_point.x() - arrow_size * math.cos(angle + math.pi/6),
            end_point.y() - arrow_size * math.sin(angle + math.pi/6)
        )
    
        arrow_path = QPainterPath()
        arrow_path.moveTo(end_point)
        arrow_path.lineTo(arrow_p1)
        arrow_path.lineTo(arrow_p2)
        arrow_path.lineTo(end_point)
    
        self.scene.addPath(arrow_path, pen, QBrush(color))

    def draw_fplc_diagram(self):
        # Draw components
        for key, comp in self.components.items():
            self.draw_3d_component(comp['pos'][0], comp['pos'][1], 
                                 comp['text'], self.colors[comp['type']])
        
        # Draw flow paths
        for start, end in self.flow_paths:
            start_pos = self.components[start]['pos']
            end_pos = self.components[end]['pos']
            self.draw_flow_line(start_pos, end_pos)
    
    def save_diagram(self):
        # Get the scene rect to determine the image size
        scene_rect = self.scene.itemsBoundingRect()
        # Add some padding
        scene_rect.adjust(-10, -10, 10, 10)
    
        # Create an image with the scene dimensions
        image = QImage(scene_rect.size().toSize(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
    
        # Create painter for the image
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
        # Render the scene onto the image
        self.scene.render(painter, QRectF(image.rect()), scene_rect)
        painter.end()
    
        # Open file dialog for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FPLC Diagram",
            "fplc_diagram.png",
            "PNG Files (*.png)"
        )
    
        if file_path:
            image.save(file_path)

class FPLCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("In-house FPLC System Flow Diagram")
        self.view = FPLCSystemView()
        self.setCentralWidget(self.view)
        self.resize(1200, 800)
        self.setStyleSheet("background-color: white;")
        
        # Create menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        # Add Save action
        save_action = file_menu.addAction("Save as PNG")
        save_action.triggered.connect(self.view.save_diagram)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FPLCWindow()
    window.show()
    sys.exit(app.exec())
