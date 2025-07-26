using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ignore_collision_between_character_to_NPC : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        // Find all GameObjects with the specified tag
        GameObject[] npcs = GameObject.FindGameObjectsWithTag("NPC");

        // Ignore collision with each NPC GameObject
        foreach (GameObject npc in npcs)
        {
            if (npc != gameObject)  // Make sure not to ignore collision with itself
            {
                Physics2D.IgnoreCollision(GetComponent<BoxCollider2D>(), npc.GetComponent<BoxCollider2D>(), true);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}

