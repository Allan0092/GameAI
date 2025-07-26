using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class music_player_loop : MonoBehaviour
{
    // Reference to the AudioSource component
    private AudioSource audioSource;

    // The AudioClip variable to hold the music
    public AudioClip musicClip;


    // Start is called before the first frame update
    void Start()
    {
        // Check if the application is playing (not in the Unity Editor)
        if (!Application.isEditor || Application.isPlaying)
        {
            // Add an AudioSource component to the GameObject
            audioSource = gameObject.AddComponent<AudioSource>();

            // Load the music into the AudioSource component
            audioSource.clip = musicClip;

            // Set the audio to loop
            audioSource.loop = true;

            // Play the music
            audioSource.Play();
        }
    }


    // Update is called once per frame
    void Update()
    {
        
    }
}

