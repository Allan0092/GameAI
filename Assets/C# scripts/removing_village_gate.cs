using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class removing_village_gate : MonoBehaviour
{
    public string GameObjectName = "bird character sprite";
    private GameObject gate;

    // Start is called before the first frame update
    void Start()
    {
        // Getting a reference of the gate GameObject
        gate = GameObject.Find(GameObjectName); 

        // Ensure a sprite is assigned in the Unity Editor
        if (gate == null)
        {
            Debug.LogError("GameObject Box Collider 2D not assigned. Please check the configuration.");
        }

    }

    // Update is called once per frame
    void Update()
    {
        // Getting a reference from the 'chief_NPC_interaction' C# script component 
        chief_NPC_interaction chiefNPCInteractionReference = GetComponent<chief_NPC_interaction>();
        
        if (chiefNPCInteractionReference.completedinteraction == true)
        {
            // Deactivating the entire gate GameObject
            gate.SetActive(false);
        }
    }
}
