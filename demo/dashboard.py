"""
Interactive Streamlit Dashboard for Identity Resolution Demo

Shows real-time probabilistic assignment of streaming sessions to household members.
Deploy to Streamlit Cloud or run locally.

Features:
- Live session assignment visualization
- Probability distributions per household member
- Ground truth vs predicted comparison
- Real-time accuracy metrics
- Interactive household configuration
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.household_inference import HouseholdInferenceEngine, ClusteringConfig
from core.probabilistic_resolver import ProbabilisticIdentityResolver
from models.streaming_event import Session, StreamingEvent

# Page configuration
st.set_page_config(
    page_title="Identity Resolution Engine Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .person-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .high-confidence {
        border-left: 4px solid #00cc00;
    }
    .medium-confidence {
        border-left: 4px solid #ffcc00;
    }
    .low-confidence {
        border-left: 4px solid #ff0000;
    }
</style>
""", unsafe_allow_html=True)


def generate_synthetic_household(
    num_persons: int = 3,
    num_sessions: int = 50,
    account_id: str = "demo_account"
) -> Tuple[List[Session], Dict]:
    """
    Generate synthetic household with ground truth labels.
    
    Returns:
        sessions: List of sessions with ground truth person IDs
        ground_truth: Dict mapping sessions to true persons
    """
    sessions = []
    ground_truth = {}
    
    # Define persona patterns
    personas = {
        0: {
            "name": "Primary Adult",
            "typical_hours": [19, 20, 21, 22],  # Evening
            "devices": ["tv", "desktop"],
            "genres": {"Drama": 0.4, "Documentary": 0.3, "Thriller": 0.3},
            "avg_duration": 7200  # 2 hours
        },
        1: {
            "name": "Teen",
            "typical_hours": [15, 16, 20, 21, 22, 23],  # After school + evening
            "devices": ["mobile", "tablet"],
            "genres": {"SciFi": 0.4, "Action": 0.3, "Comedy": 0.3},
            "avg_duration": 3600  # 1 hour
        },
        2: {
            "name": "Child",
            "typical_hours": [16, 17, 18, 19],  # After school
            "devices": ["tablet"],
            "genres": {"Animation": 0.5, "Kids": 0.3, "Comedy": 0.2},
            "avg_duration": 1800  # 30 minutes
        }
    }
    
    # Generate sessions for each person
    sessions_per_person = num_sessions // num_persons
    
    for person_id in range(min(num_persons, 3)):
        persona = personas[person_id]
        
        for i in range(sessions_per_person):
            # Generate realistic timestamp
            hour = np.random.choice(persona["typical_hours"])
            day_offset = np.random.randint(0, 30)  # Last 30 days
            timestamp = datetime.now() - timedelta(days=day_offset, hours=np.random.randint(0, 24))
            timestamp = timestamp.replace(hour=hour, minute=np.random.randint(0, 60))
            
            # Generate session
            session = Session(
                session_id=f"sess_{account_id}_p{person_id}_{i}",
                account_id=account_id,
                device_fingerprint=f"device_{np.random.choice(persona['devices'])}",
                device_type=np.random.choice(persona["devices"]),
                start_time=timestamp,
                total_duration=persona["avg_duration"] + np.random.normal(0, 300),
                genres_watched={
                    genre: time_spent * persona["avg_duration"]
                    for genre, time_spent in persona["genres"].items()
                },
                event_count=np.random.poisson(30)
            )
            
            sessions.append(session)
            ground_truth[session.session_id] = {
                "person_id": person_id,
                "persona": persona["name"]
            }
    
    # Shuffle sessions
    np.random.shuffle(sessions)
    
    return sessions, ground_truth


def visualize_household_structure(
    engine: HouseholdInferenceEngine,
    sessions: List[Session],
    ground_truth: Dict
):
    """Visualize detected household members and their characteristics."""
    
    st.subheader("🏠 Household Structure Analysis")
    
    # Analyze household
    household = engine.analyze_household(sessions)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Detected Members",
            len(household.members),
            f"{household.size_confidence:.0%} confidence"
        )
    
    with col2:
        st.metric(
            "Total Sessions",
            len(sessions),
            f"{len(sessions) / max(1, len(household.members)):.0f} per person"
        )
    
    with col3:
        st.metric(
            "Silhouette Score",
            f"{household.size_confidence:.2f}",
            "Clustering quality"
        )
    
    # Show each detected person
    st.markdown("### 👤 Detected Household Members")
    
    for i, member in enumerate(household.members):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{member.label}** ({member.persona_type})")
                st.caption(f"ID: {member.person_id[:8]}...")
            
            with col2:
                st.markdown(f"📺 **{member.session_count}** sessions")
                hours = member.total_viewing_time / 3600
                st.caption(f"{hours:.1f} hours total")
            
            with col3:
                if member.primary_device_type:
                    st.markdown(f"💻 **{member.primary_device_type.title()}**")
                if member.top_genres:
                    st.caption(f"Likes: {', '.join(member.top_genres[:2])}")
            
            with col4:
                # Time pattern visualization
                if member.typical_hours:
                    peak_hour = member.typical_hours[0]
                    time_str = f"{peak_hour}:00"
                    st.markdown(f"🕐 **Peak: {time_str}**")
            
            st.markdown("---")


def visualize_probabilistic_assignment(
    engine: HouseholdInferenceEngine,
    sessions: List[Session],
    ground_truth: Dict
):
    """Show real-time probabilistic assignment with confidence scores."""
    
    st.subheader("🎯 Probabilistic Session Assignment")
    
    # Let user select a session to analyze
    session_options = [
        f"{s.session_id} ({s.start_time.strftime('%m/%d %H:%M')} - {s.device_type})"
        for s in sessions[:20]  # Show first 20 for performance
    ]
    
    selected_idx = st.selectbox(
        "Select a session to analyze assignment:",
        range(len(session_options)),
        format_func=lambda i: session_options[i]
    )
    
    selected_session = sessions[selected_idx]
    account_id = selected_session.account_id
    
    # Get assignment probabilities
    with st.spinner("Computing probabilistic assignment..."):
        # First ensure household is analyzed
        household = engine.analyze_household([s for s in sessions if s.account_id == account_id])
        
        # Get assignment
        probabilities = engine.assign_session_to_person(selected_session, account_id)
    
    # Display results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Probability Distribution")
        
        if probabilities:
            # Create bar chart of probabilities
            probs_df = pd.DataFrame([
                {"Person": pid.split("_")[-1].replace("person", "Person "),
                 "Probability": prob}
                for pid, prob in probabilities.items()
            ])
            
            fig = px.bar(
                probs_df,
                x="Person",
                y="Probability",
                color="Probability",
                color_continuous_scale="RdYlGn",
                range_y=[0, 1]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No clustering model available for this account yet")
    
    with col2:
        st.markdown("#### Assignment Details")
        
        # Show ground truth
        true_info = ground_truth.get(selected_session.session_id, {})
        if true_info:
            st.markdown(f"**Ground Truth:** {true_info['persona']}")
        
        # Show predicted
        if probabilities:
            predicted = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted]
            
            st.markdown(f"**Predicted:** Person {predicted.split('_')[-1]}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            
            # Color-coded confidence
            if confidence >= 0.8:
                st.success("High Confidence ✓")
            elif confidence >= 0.5:
                st.warning("Medium Confidence ⚠")
            else:
                st.error("Low Confidence ✗")
        
        # Session features
        st.markdown("#### Session Features")
        st.caption(f"Device: {selected_session.device_type}")
        st.caption(f"Duration: {selected_session.total_duration/60:.0f} minutes")
        st.caption(f"Events: {selected_session.event_count}")


def visualize_accuracy_metrics(
    engine: HouseholdInferenceEngine,
    sessions: List[Session],
    ground_truth: Dict
):
    """Calculate and display accuracy metrics vs ground truth."""
    
    st.subheader("📊 Accuracy Analysis")
    
    # Group sessions by account
    accounts = {}
    for session in sessions:
        if session.account_id not in accounts:
            accounts[session.account_id] = []
        accounts[session.account_id].append(session)
    
    # Calculate accuracy
    correct = 0
    total = 0
    confidences = []
    
    accuracy_by_persona = {}
    
    with st.spinner("Computing accuracy metrics..."):
        for account_id, account_sessions in accounts.items():
            # Analyze household
            household = engine.analyze_household(account_sessions)
            
            for session in account_sessions:
                true_info = ground_truth.get(session.session_id, {})
                if not true_info:
                    continue
                
                # Get prediction
                probs = engine.assign_session_to_person(session, account_id)
                if not probs:
                    continue
                
                predicted = max(probs, key=probs.get)
                confidence = probs[predicted]
                
                # Map predicted to person index
                pred_idx = int(predicted.split("_")[-1])
                true_idx = true_info["person_id"]
                
                is_correct = pred_idx == true_idx
                
                if is_correct:
                    correct += 1
                total += 1
                confidences.append(confidence)
                
                # Track by persona
                persona = true_info["persona"]
                if persona not in accuracy_by_persona:
                    accuracy_by_persona[persona] = {"correct": 0, "total": 0}
                accuracy_by_persona[persona]["total"] += 1
                if is_correct:
                    accuracy_by_persona[persona]["correct"] += 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = correct / total if total > 0 else 0
        st.metric(
            "Overall Accuracy",
            f"{accuracy:.1%}",
            f"{correct}/{total} correct"
        )
    
    with col2:
        avg_conf = np.mean(confidences) if confidences else 0
        st.metric(
            "Avg Confidence",
            f"{avg_conf:.1%}",
            "Well-calibrated" if 0.6 < avg_conf < 0.9 else "Review calibration"
        )
    
    with col3:
        st.metric(
            "Households Analyzed",
            len(accounts),
            f"{len(sessions)} total sessions"
        )
    
    with col4:
        if confidences:
            brier = np.mean([(c - (1 if c > 0.5 else 0))**2 for c in confidences])
            st.metric(
                "Brier Score",
                f"{brier:.3f}",
                "Good" if brier < 0.15 else "Fair" if brier < 0.25 else "Poor"
            )
    
    # Accuracy by persona
    if accuracy_by_persona:
        st.markdown("### Accuracy by Persona Type")
        
        persona_data = []
        for persona, stats in accuracy_by_persona.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            persona_data.append({
                "Persona": persona,
                "Accuracy": acc,
                "Count": stats["total"]
            })
        
        persona_df = pd.DataFrame(persona_data)
        
        fig = px.bar(
            persona_df,
            x="Persona",
            y="Accuracy",
            color="Accuracy",
            text="Count",
            color_continuous_scale="RdYlGn",
            range_y=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("<h1 class='main-header'>🔍 Identity Resolution Engine</h1>", 
                unsafe_allow_html=True)
    st.markdown("*Real-time probabilistic assignment of streaming sessions to household members*")
    
    # Sidebar controls
    st.sidebar.markdown("## Configuration")
    
    num_persons = st.sidebar.slider("Household Size", 1, 6, 3,
                                    help="Number of distinct people in the household")
    
    num_sessions = st.sidebar.slider("Total Sessions", 10, 200, 50,
                                     help="Total viewing sessions to analyze")
    
    show_ground_truth = st.sidebar.checkbox("Show Ground Truth", True,
                                           help="Display actual person labels for comparison")
    
    # Generate data button
    if st.sidebar.button("🔄 Generate New Household", type="primary"):
        st.session_state.sessions = None
        st.experimental_rerun()
    
    # Initialize or retrieve sessions
    if "sessions" not in st.session_state or st.session_state.sessions is None:
        with st.spinner("Generating synthetic household data..."):
            sessions, ground_truth = generate_synthetic_household(
                num_persons=num_persons,
                num_sessions=num_sessions
            )
            st.session_state.sessions = sessions
            st.session_state.ground_truth = ground_truth
    
    sessions = st.session_state.sessions
    ground_truth = st.session_state.ground_truth
    
    # Initialize engine
    engine = HouseholdInferenceEngine(config=ClusteringConfig())
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "🏠 Household Structure",
        "🎯 Live Assignment",
        "📊 Accuracy Metrics"
    ])
    
    with tab1:
        visualize_household_structure(engine, sessions, ground_truth)
    
    with tab2:
        visualize_probabilistic_assignment(engine, sessions, ground_truth)
    
    with tab3:
        if show_ground_truth:
            visualize_accuracy_metrics(engine, sessions, ground_truth)
        else:
            st.info("Enable 'Show Ground Truth' in sidebar to see accuracy metrics")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 | MIT License")
    st.sidebar.caption("GitHub: Michaelrobins938/probabilistic-identity-resolution")


if __name__ == "__main__":
    main()
