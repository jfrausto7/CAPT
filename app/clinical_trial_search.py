import requests
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StudyLocation:
    facility: str
    city: str
    state: str
    country: str
    status: str
    contacts: List[Dict[str, str]]

@dataclass
class ClinicalTrial:
    nct_id: str
    title: str
    status: str
    start_date: str
    completion_date: str
    conditions: List[str]
    locations: List[StudyLocation]
    brief_summary: str
    sponsor: str
    phase: Optional[List[str]]
    
    @property
    def study_url(self) -> str:
        """Returns the full URL to view the study on ClinicalTrials.gov"""
        return f"https://clinicaltrials.gov/study/{self.nct_id}"
    
    @property
    def study_api_url(self) -> str:
        """Returns the API URL for the study"""
        return f"https://clinicaltrials.gov/api/v2/studies/{self.nct_id}"

def search_clinical_trials(
    drug: str,
    location: str,
    condition: Optional[str] = None,
    status: str = "RECRUITING"
) -> List[ClinicalTrial]:
    """
    Search ClinicalTrials.gov API for studies based on drug, location, and optional condition.
    
    Args:
        drug (str): Drug name (e.g., "MDMA", "LSD", "Psilocybin")
        location (str): Location name (e.g., "California", "New York")
        condition (str, optional): Medical condition (e.g., "PTSD", "Depression")
        status (str, optional): Study status (default: "RECRUITING")
    
    Returns:
        List[ClinicalTrial]: List of parsed clinical trial data objects
        
    Raises:
        requests.RequestException: If the API request fails
    """
    # Base URL for the API
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Build query parameters
    params = {
        "query.titles": drug,
        "query.locn": location,
        "filter.overallStatus": status,
        "format": "json"
    }
    
    # Add condition if provided
    if condition:
        params["query.cond"] = condition
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        trials = []
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            
            # Extract locations
            locations = []
            locations_data = protocol.get("contactsLocationsModule", {}).get("locations", [])
            for loc in locations_data:
                locations.append(StudyLocation(
                    facility=loc.get("facility", ""),
                    city=loc.get("city", ""),
                    state=loc.get("state", ""),
                    country=loc.get("country", ""),
                    status=loc.get("status", ""),
                    contacts=loc.get("contacts", [])
                ))
            
            # Create ClinicalTrial object
            trial = ClinicalTrial(
                nct_id=protocol.get("identificationModule", {}).get("nctId", ""),
                title=protocol.get("identificationModule", {}).get("briefTitle", ""),
                status=protocol.get("statusModule", {}).get("overallStatus", ""),
                start_date=protocol.get("statusModule", {}).get("startDateStruct", {}).get("date", ""),
                completion_date=protocol.get("statusModule", {}).get("completionDateStruct", {}).get("date", ""),
                conditions=protocol.get("conditionsModule", {}).get("conditions", []),
                locations=locations,
                brief_summary=protocol.get("descriptionModule", {}).get("briefSummary", ""),
                sponsor=protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
                phase=protocol.get("designModule", {}).get("phases", [])
            )
            trials.append(trial)
            
        return trials
        
    except requests.RequestException as e:
        print(f"Error making API request: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    try:
        # Search for MDMA trials in California for PTSD
        trials = search_clinical_trials("MDMA", "United States")
        
        # Print results in a readable format
        for trial in trials:
            print(f"\nStudy: {trial.title}")
            print(f"NCT ID: {trial.nct_id}")
            print(f"Status: {trial.status}")
            print(f"Phase: {', '.join(trial.phase) if trial.phase else 'Not specified'}")
            print(f"Conditions: {', '.join(trial.conditions)}")
            print(f"Start Date: {trial.start_date}")
            print(f"Completion Date: {trial.completion_date}")
            print(f"Sponsor: {trial.sponsor}")
            print(f"Study URL: {trial.study_url}")
            print("\nBrief Summary:")
            print(trial.brief_summary)
            print("\nLocations:")
            for loc in trial.locations:
                print(f"- {loc.facility} ({loc.city}, {loc.state}, {loc.country})")
                for contact in loc.contacts:
                    if contact.get("role") == "PRINCIPAL_INVESTIGATOR":
                        print(f"  PI: {contact.get('name')}")
            print("-" * 80)
            
    except Exception as e:
        print(f"An error occurred: {e}")