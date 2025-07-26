using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class changing_sprite_image : MonoBehaviour
{
    public string playerGameObjectName = "bird character sprite";
    private GameObject player;
    public Sprite newSprite; // Assign the new sprite in the Unity Editor

    private SpriteRenderer playerspriteRendererComponent;


    // Start is called before the first frame update
    void Start()
    {
        // Getting a reference of the player character GameObject
        player = GameObject.Find(playerGameObjectName); 

        // Get the player's 'SpriteRenderer' component
        playerspriteRendererComponent = player.GetComponent<SpriteRenderer>();

        // Ensure a sprite is assigned in the Unity Editor
        if (playerspriteRendererComponent == null || newSprite == null)
        {
            Debug.LogError("SpriteRenderer or newSprite not assigned. Please check the configuration.");
        }

    }

    // Update is called once per frame
    void Update()
    {
        // Getting a reference from the 'chief_NPC_interaction' C# script component 
        chief_NPC_interaction chiefNPCInteractionReference = GetComponent<chief_NPC_interaction>();
        
        if (chiefNPCInteractionReference.completedinteraction == true)
        {
            // Change the sprite
            playerspriteRendererComponent.sprite = newSprite;
        }
    }
}
