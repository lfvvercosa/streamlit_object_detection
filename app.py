import streamlit as st
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import os
import colorsys
from collections import Counter

# Fixed directories
IMAGES_DIR = "test_images"
PREDICTIONS_DIR = "predictions"

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = tuple(int(255 * c) for c in rgb)
        colors.append(color)
    return colors

def parse_xml(xml_path):
    """Parse XML file and extract bounding box information"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        confidence = float(obj.find('confidence').text)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'confidence': confidence,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    
    return objects

def zoom_image(image, zoom_factor, center_x=0.5, center_y=0.5):
    """Zoom into an image around a center point"""
    if zoom_factor == 1.0:
        return image
    
    height, width = image.shape[:2]
    
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)
    
    start_x = int(width * center_x - new_width / 2)
    start_y = int(height * center_y - new_height / 2)
    
    start_x = max(0, min(start_x, width - new_width))
    start_y = max(0, min(start_y, height - new_height))
    
    cropped = image[start_y:start_y + new_height, start_x:start_x + new_width]
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return zoomed

def main():
    st.set_page_config(layout="wide", page_title="Object Detection Visualization")

    # Initialize session state
    if 'zoom_factor' not in st.session_state:
        st.session_state.zoom_factor = 1.0
    if 'center_x' not in st.session_state:
        st.session_state.center_x = 0.5
    if 'center_y' not in st.session_state:
        st.session_state.center_y = 0.5
    if 'show_predictions' not in st.session_state:
        st.session_state.show_predictions = False
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'predictions_loaded' not in st.session_state:
        st.session_state.predictions_loaded = False

    # Sidebar configuration
    with st.sidebar:
        st.header("Controls")
        
        # Zoom controls in sidebar
        st.subheader("🔍 Zoom Controls")
        
        st.session_state.zoom_factor = st.slider(
            "Zoom Level",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.zoom_factor,
            step=0.1
        )
        
        st.session_state.center_x = st.slider(
            "Horizontal Position",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.center_x,
            step=0.01
        )
        
        st.session_state.center_y = st.slider(
            "Vertical Position",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.center_y,
            step=0.01
        )
    
    # Load custom CSS
    load_css('styles.css')
    
    
    
    # Get list of images
    image_files = [f for f in os.listdir(IMAGES_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Layout with sidebar (gallery) and main content
    col_gallery, col_main = st.columns([1.5, 4])  
    
    with col_gallery:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown("""
            <div class="header">
                <h3>📸 Test Images</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for image in image_files:
            st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
            image_path = os.path.join(IMAGES_DIR, image)
            thumbnail = Image.open(image_path)
            thumbnail.thumbnail((200, 200))
            
            # Image first
            st.image(thumbnail, use_container_width=True)
            
            # Button with "Selecionar" text
            if st.button(
                "Selecionar",  # Changed to "Selecionar"
                key=f"thumb_{image}",
                help=f"Select {image}",
                use_container_width=True
            ):
                st.session_state.selected_image = image
                st.session_state.show_predictions = False
                
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    with col_main:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        
        # Header
        st.markdown("""
            <div class="header">
                <h3>🟢 Object Detection Visualization</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.selected_image:
            # Load image
            image_path = os.path.join(IMAGES_DIR, st.session_state.selected_image)
            image = Image.open(image_path)
            img_array = np.array(image)
            
            # Get corresponding XML path
            base_name = os.path.splitext(st.session_state.selected_image)[0]
            xml_path = os.path.join(PREDICTIONS_DIR, f"{base_name}.xml")
            
            # Control panel
            st.markdown('<div class="control-panel-buttons">', unsafe_allow_html=True)
            
            # Create a container for the buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Load Predictions button - always visible but disabled when predictions are loaded
                if st.button("🎯 Load Predictions", 
                         type="primary", 
                         key="load_pred",
                         disabled=st.session_state.show_predictions,
                         use_container_width=True):
                    st.session_state.show_predictions = True
                    st.session_state.predictions_loaded = True
                    st.rerun()
            
            with col2:
                # Reset View button - will now clear predictions
                if st.button("↺ Reset View", 
                         key="reset_view",
                         use_container_width=True):
                    st.session_state.zoom_factor = 1.0
                    st.session_state.center_x = 0.5
                    st.session_state.center_y = 0.5
                    st.session_state.show_predictions = False
                    st.session_state.predictions_loaded = False
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.show_predictions and os.path.exists(xml_path):
                objects = parse_xml(xml_path)
                unique_labels = list(set(obj['name'] for obj in objects))
                
                # Custom color mapping for specific classes
                color_map = {}
                remaining_labels = []
                
                # Set predefined colors for specific classes
                for label in unique_labels:
                    if label == "on_green":
                        color_map[label] = (51, 102, 0)  # Blue
                    elif label == "off":
                        color_map[label] = (255, 255, 255)  # Red
                    else:
                        remaining_labels.append(label)
                
                # Generate colors for any remaining labels
                if remaining_labels:
                    remaining_colors = generate_distinct_colors(len(remaining_labels))
                    for label, color in zip(remaining_labels, remaining_colors):
                        color_map[label] = color
                
                img_draw = img_array.copy()
                for obj in objects:
                    label = obj['name']
                    xmin, ymin, xmax, ymax = obj['bbox']
                    color = color_map[label]
                    cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 2)
                
                display_image = zoom_image(
                    img_draw,
                    st.session_state.zoom_factor,
                    st.session_state.center_x,
                    st.session_state.center_y
                )
                
                # Display image and legend
                st.image(display_image, use_container_width=True)
                
                # Legend display code stays the same
                class_counts = Counter(obj['name'] for obj in objects)
                with st.sidebar:
                    st.markdown("### Legend")
                    for label, color in color_map.items():
                        count = class_counts[label]
                        avg_conf = np.mean([obj['confidence'] for obj in objects if obj['name'] == label])
                        
                        st.markdown(f"""
                            <div class="legend-item" style="background-color: #1E1E1E; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                <div style="display: inline-block; width: 20px; height: 20px; background-color: rgb{color}; border: 1px solid #666; margin-right: 10px;"></div>
                                <div style="display: inline-block;">
                                    <strong>{label}</strong><br>
                                    Count: {count}<br>
                                    Confidence: {avg_conf:.2f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f'<div style="color: white;"><strong>Total Objects:</strong> {len(objects)}</div>', unsafe_allow_html=True)
            else:
                display_image = zoom_image(
                    img_array,
                    st.session_state.zoom_factor,
                    st.session_state.center_x,
                    st.session_state.center_y
                )
                st.image(display_image, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Select an image from the gallery to begin")
        
        st.markdown('</div>', unsafe_allow_html=True)


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    import base64
    from io import BytesIO
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


if __name__ == '__main__':
    main()